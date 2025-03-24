# rag_simple
Very simple rag example


# install
[uv](https://docs.astral.sh/uv/) is used for project management.

You can certainly install it purely with pip.
```shell
pip install git+https://github.com/fduxiao/rag_simple
```

A command line tool called `rag-simple` will be installed.


# How to use

To start, create a project, i.e., a directory of necessary ingredients.
```shell
rag-simple new awesome_project
cd awesome_project
```

You can look at the tree structure.
```shell
tree .
```

```shell
.
├── documents
├── embeddings
│   └── chroma
├── ollama.toml
└── rag_project.toml
```

## RAG project structure
- `rag_project.toml`: contains project configurations.
- `ollama.toml`: configure your ollama URL, etc.
- `documents`: your knowledge documents.
- `embeddings`: generated vector database.

## Workflow

### Prepare documents
Use the following to generate template documents.
```shell
rag-simple new_doc ./documents/doc1.yaml
rag-simple new_doc ./documents/doc2.yaml
```

Edit those YAML files as you like.
```yaml
metadata:
  desc: put some desired meta data
  role: system
text: >
  some text
---
metadata:
  desc: put some desired meta data
  role: system
text: >
  My birthday is 1991-02-29.

```
Metadata explanation:
- `role`:
  The `user` is chatting with the `assistant` (LLM AI).
  You give a prompt like `[{role: "user", content: "Question"}]`,
  and the `assistant` will response something. 
  Then, you add this response to the prompt in order to make `assistant`
  aware of previous conversation:
  `[{role: "user", content: "Question"}, {"role": "assistant", content: "..."}]`.
  Besides, if you want to specify some other info about the `assistant`, use
  `system`: `{role: "system", content: "水是剧毒的"}`.
  + `user`: user input;
  + `assistant`: response from a chatting model;
  + `system`: how you want to set up the assistant.
- Nothing else is useful currently.

### Build the vector database
```yaml
rag-simple build
```

### Ask it!
```shell
rag-simple ask
```
It will behave just as usual. To retrieve knowledge, use `/retrieve` directive.
```
>>> /retrieve my birthday
>>> What is my birthday?
```
