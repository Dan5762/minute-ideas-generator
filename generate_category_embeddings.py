import json

# Load the ideas from the file
with open('ideas.json') as file:
    data = json.load(file)

# Create a dictionary to group the ideas by source
grouped_ideas = {}

# Iterate over the dictionaries in the data
for idea in data:
    # Get the source field value
    source = idea.get('source')

    # Check if the source field exists
    if source:
        # Check if the source is already a key in the grouped_ideas dictionary
        if source in grouped_ideas:
            # Append the idea to the existing list of ideas for the source
            grouped_ideas[source].append(idea)
        else:
            # Create a new list with the idea and assign it to the source key
            grouped_ideas[source] = [idea]

# Create a dictionary to store the idea embeddings
idea_embeddings = {}
for source, ideas in grouped_ideas.items():
    # Get the category name from the source URL
    category_name = source.split('/')[-1].replace('_', ' ')

    # Average the embeddings of the ideas
    embedding = [0] * 768
    for idea in ideas:
        embedding = [sum(x) for x in zip(embedding, idea['embedding'])]
    embedding = [x / len(ideas) for x in embedding]

    # Round the embedding values to 4 decimal places
    embedding = [round(x, 4) for x in embedding]

    idea_embeddings[category_name] = embedding

# Save the idea embeddings to a file
with open('categories.json', 'w') as file:
    json.dump(idea_embeddings, file)
