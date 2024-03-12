import os
import json

import wikipediaapi
from openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from pydantic.fields import Field
from sentence_transformers import SentenceTransformer

os.environ["OPENAI_API_KEY"] = os.environ["MINUTE_IDEAS_OPENAI_API_KEY"]
client = OpenAI()

wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='MinuteTrivia (danlong1998@icloud.com)',
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)


class Question(BaseModel):
    question: str = Field(
        description="The question, either true or false or multiple choice")
    answer: int = Field(
        description="The index of the answer to the question (indexes start from 0)")
    choices: list[str] = Field(description="The choices for the question")


class Trivia(BaseModel):
    title: str = Field(description="The title of the trivia")
    content: str = Field(
        description="A brief interesting bit of trivia from the text, only a couple of sentences")
    questions: list[Question] = Field(
        description="A list of 3 questions that can be answered with the extracted trivia")


class Trivias(BaseModel):
    trivias: list[Trivia] = Field(description="The list of trivias")


def get_embeddings(text):
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embedding = model.encode(text).tolist()
    embedding = [round(x, 4) for x in embedding]

    return embedding


def parse_output(output, parser):
    from langchain.output_parsers import OutputFixingParser
    from langchain.schema import OutputParserException
    from langchain_community.chat_models import ChatOpenAI

    try:
        output_obj = parser.parse(output)
    except OutputParserException:
        gpt_4_turbo = ChatOpenAI(model_name="gpt-4-1106-preview")
        new_parser = OutputFixingParser.from_llm(
            parser=parser, llm=gpt_4_turbo)
        output_obj = new_parser.parse(output)

    return output_obj


def get_page_trivia(page):
    excluded_sections = ["See also", "Sources", "Notes",
                         "References", "Further reading", "External links"]

    trivias = []
    print()
    print(page.title)
    for section in page.sections:
        print(section.title)

        if section.title in excluded_sections or section.text == "":
            continue

        parser = PydanticOutputParser(pydantic_object=Trivias)
        prompt = PromptTemplate(
            template="""Extract useful information from the following text.

Text: {section_text}

{format_instructions}""",
            input_variables=["section_text"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()},
        )
        _input = prompt.format_prompt(section_text=section.text)

        chat_completion = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "user",
                    "content": _input.to_string()
                }
            ],
            temperature=0,
            max_tokens=1000
        )

        model_output = chat_completion.choices[0].message.content

        new_trivias = parse_output(model_output, parser).dict()['trivias']

        category_name = page.title.replace(
            '_', ' ') + ' - ' + section.title.replace('_', ' ')

        new_trivias = [
            dict(
                trivia,
                read=None,
                source=page.fullurl,
                embedding=get_embeddings(trivia['content']),
                uuid=f"{category_name} - {i}"
            ) for i, trivia in enumerate(new_trivias)
        ]

        trivias += new_trivias

    return trivias


if __name__ == "__main__":
    pages = ["Political_philosophy", "Physics",
             "Agriculture_in_the_Middle_Ages"]
    page_trivias = {}

    with open("ideas.json", "r") as f:
        page_trivias = json.load(f)

    for page_name in pages:
        page = wiki_wiki.page(page_name)

        trivias = get_page_trivia(page)
        page_trivias += trivias

        with open("ideas.json", "w") as f:
            f.write(json.dumps(page_trivias))
