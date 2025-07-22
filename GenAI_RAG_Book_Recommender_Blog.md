## Lesson 2: Your RAG Prototype


In this lesson, you will build a RAG prototype in this notebook which you will learn how to automate in the next lesson. You are provided with text files containing book descriptions. You will create embeddings based on the book description and store them in a vector database. Here's what you will do:
- read book descriptions from the text files stored under `include/data`
- use `fastembed` to create the vector embedding for each book description
- store the embeddings and the book metadata in a local `weaviate` database

### Import libraries

```python
import os
import json
from IPython.display import JSON

from fastembed import TextEmbedding

import weaviate
from weaviate.classes.data import DataObject

from helper import suppress_output
```

```python
# Warning control
import warnings
warnings.filterwarnings('ignore')
```

<p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>To access <code>requirements.txt</code> and <code>helper.py</code> files, and <code>include</code> folder:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook, 2) click on <em>"Open"</em> and then 3) click on <em>"L2"</em>. For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>

### Set variables

```python
COLLECTION_NAME = "Books"  # capitalize the first letter of collection names
BOOK_DESCRIPTION_FOLDER = "include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
```

Note regarding the variable `COLLECTION_NAME`: Weaviate stores data in ["collections"](https://weaviate.io/developers/academy/py/starter_text_data/text_collections/create_collection). A collection is a set of objects that share the same data structure. In the Weaviate instance of this lesson, you will create a collection of books. Each book object will have a vector embedding and a set of properties.

### Instantiate Embedded Weaviate client

You will now create a local Weaviate instance: [Embedded Weaviate](https://weaviate.io/developers/weaviate/connections/connect-embedded), which is a way to run a Weaviate instance from your application code rather than from a stand-alone Weaviate server installation. 

In the next lessons, you will be interacting with the latter option; you'll be provided with a Weaviate instance running in a [Docker](https://docs.docker.com/) container.

```python
with suppress_output():
    client = weaviate.connect_to_embedded(
        persistence_data_path= "tmp/weaviate",
    )
print("Started new embedded Weaviate instance.")
print(f"Client is ready: {client.is_ready()}")
```

### Create the collection 

You will now create the Books collection inside the Weaviate instance.

```python
existing_collections = client.collections.list_all()
existing_collection_names = existing_collections.keys()

if COLLECTION_NAME not in existing_collection_names:
    print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
    collection = client.collections.create(name=COLLECTION_NAME)
    print(f"Collection {COLLECTION_NAME} created successfully.")
else:
    print(f"Collection {COLLECTION_NAME} already exists. No action taken.")
    collection = client.collections.get(COLLECTION_NAME)
```

### Extract text from local files

You are provided with the `BOOK_DESCRIPTION_FOLDER` (`include/data`) inside the L2 directory. It contains some text files, where each text file contains some book descriptions. You'll now list the text files to discover how many of such files you are provided. 

```python
# list the book description files
book_description_files = [
    f for f in os.listdir(BOOK_DESCRIPTION_FOLDER)
    if f.endswith('.txt')
]

print(f"The following files with book descriptions were found: {book_description_files}")
```

You'll add another file that contains some additional book descriptions. Feel free to add your own book description file. 

```python
# Add your own book description file
# Format 
# [Integer Index] ::: [Book Title] ([Release year]) ::: [Author] ::: [Description]

my_book_description = """0 ::: The Idea of the World (2019) ::: Bernardo Kastrup ::: An ontological thesis arguing for the primacy of mind over matter.
1 ::: Exploring the World of Lucid Dreaming (1990) ::: Stephen LaBerge ::: A practical guide to learning and enjoying lucid dreams.
"""

# Write to file
with open(f"{BOOK_DESCRIPTION_FOLDER}/my_book_descriptions.txt", 'w') as f:
    f.write(my_book_description)
```

You'll now loop through each text file. For each text file, you will read each line, which corresponds to one book, to extract the title, author and text description of that book. You will save the data in a list of Python dictionaries, where each dictionary corresponds to one book.   

```python
book_description_files = [
    f for f in os.listdir(BOOK_DESCRIPTION_FOLDER)
    if f.endswith('.txt')
]

list_of_book_data = []

for book_description_file in book_description_files:
    with open(
        os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
    ) as f:
        book_descriptions = f.readlines()
    
    titles = [
        book_description.split(":::")[1].strip()
        for book_description in book_descriptions
    ]
    authors = [
        book_description.split(":::")[2].strip()
        for book_description in book_descriptions
    ]
    book_description_text = [
        book_description.split(":::")[3].strip()
        for book_description in book_descriptions
    ]
    
    book_descriptions = [
        {
            "title": title,
            "author": author,
            "description": description,
        }
        for title, author, description in zip(
            titles, authors, book_description_text
        )
    ]

    list_of_book_data.append(book_descriptions)
```

```python
JSON(json.dumps(list_of_book_data))
```

### Create vector embeddings from descriptions

For each book in the list of book data you extracted, you will now create an embedding vector based on the text description. You will store the the vector of embeddings in the list `list_of_description_embeddings`.

```python
embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)  

list_of_description_embeddings = []

for book_data in list_of_book_data:
    book_descriptions = [book["description"] for book in book_data]
    description_embeddings = [
        list(embedding_model.embed([desc]))[0] for desc in book_descriptions
    ]

    list_of_description_embeddings.append(description_embeddings)
```

### Load embeddings to Weaviate

In the books collection of Weaviate, you will create an item for each data object (book). The item has two attributes:
- `vector`: which represents the vector embedding of the book text description
- `properties`: which is a python dictionary that contains the book metadata: title, author and text description.

```python
for book_data_list, emb_list in zip(list_of_book_data, list_of_description_embeddings):
    items = []
    
    for book_data, emb in zip(book_data_list, emb_list):
        item = DataObject(
            properties={
                "title": book_data["title"],
                "author": book_data["author"],
                "description": book_data["description"],
            },
            vector=emb
        )
        items.append(item)
    
    collection.data.insert_many(items)
```

### Query for a book recommendation using semantic search

Now that you have the embeddings stored in the Weaviate instance, you can query the vector database. You are provided with a query that you will first map it to its embedding vector. You will then pass this vector embedding to the method: `query.near_vector` of the Weaviate `Books` collection. 

```python
query_str = "A philosophical book"

embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)  
collection = client.collections.get(COLLECTION_NAME)

query_emb = list(embedding_model.embed([query_str]))[0]

results = collection.query.near_vector(
    near_vector=query_emb,
    limit=1,
)
for result in results.objects:
    print(f"You should read: {result.properties['title']} by {result.properties['author']}")
    print("Description:")
    print(result.properties["description"])
```

### Optional Cleanup utilities

These are optional cleanup utilities that you can locally use to remove the custom book description file, a collection in weaviate or the entire Weaviate instance.

```python
# ## Remove a book description file

# import os

# file_path = f"{BOOK_DESCRIPTION_FOLDER}/my_book_descriptions.txt"

# # Remove the file
# if os.path.exists(file_path):
#     os.remove(file_path)
# else:
#     print(f"File not found: {file_path}")
```

```python
# ## Remove a collection from an existing Weaviate instance

# client.collections.delete(COLLECTION_NAME)
```

```python
# ## Delete a Weaviate instance
# ## This cell can take a few seconds to run  

# import shutil

# client.close()

# EMBEDDED_WEAVIATE_PERSISTENCE_PATH = "tmp/weaviate"

# if os.path.exists(EMBEDDED_WEAVIATE_PERSISTENCE_PATH):
#     shutil.rmtree(EMBEDDED_WEAVIATE_PERSISTENCE_PATH)
#     if not os.path.exists(EMBEDDED_WEAVIATE_PERSISTENCE_PATH):
#         print(f"Verified: '{EMBEDDED_WEAVIATE_PERSISTENCE_PATH}' no longer exists.")
#         print(f"Weaviate embedded data at '{EMBEDDED_WEAVIATE_PERSISTENCE_PATH}' deleted.")
```

### Resources

- [Weaviate Docs](https://weaviate.io/developers/weaviate)
- [What is FastEmbed?](https://qdrant.github.io/fastembed/)
- [Weaviate Short Course - Vector Databases: from Embeddings to Applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)
- [Weaviate Short Course - Building Multimodal Search and RAG](https://www.deeplearning.ai/short-courses/building-multimodal-search-and-rag/)

<div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
<p> 💻 &nbsp; <b>To access <code>requirements.txt</code> and <code>helper.py</code> files and <code> include</code> folder:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook 2) click on <em>"Open"</em> and then 3) click on <em>"L2"</em>.

<p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>

<p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>

</div>

## Lesson 3: Building a Simple Pipeline

Before you transform your RAG prototype into an automated pipeline, you will learn some basic Airflow syntax.

### 3.1. Airflow UI

You will use the Airflow UI to visualize the dags, track their status and trigger them manually. Run the cell below to get the link to your Airflow UI. If asked for username and password, type `airflow` for both.

```python
import os
airflow_ui = os.environ.get('DLAI_LOCAL_URL').format(port=8080)
airflow_ui #username:airflow password:airflow (if asked)
```

### 3.2. Airflow Components - Optional Reading

You've already seen one of the Airflow components which is the Airflow UI hosted on the API server which is shown in the diagram below. Airflow has other components that interact all together to process and run the dags you write. In this course, the components are already set up for you (each component is running in a docker container). If you'd like to know how to install Airflow locally on your machine, please check the resource section below in this notebook. 

<img src="airflow_architecture_3.png" width="400">

You will write your dags as python files and save them in a dags folder. Once you add a new dag to your Airflow environment:
1. The dag processor parses your dag and stores a serialized version of the dag in the Airflow metadata database.
2. The scheduler checks the serialized dags to determine whether any dag is eligible for execution based on its defined schedule.
3. The tasks are then scheduled and subsequently queued. The workers poll the queue for any queued task instances they can run.
4. The worker who picked up the task instance runs it, and metadata such as the task instance status is sent from the worker via the API server to be stored in the Airflow metadata database. 
5. Some of this information, such as the task instance status, is in turn important for the scheduler. It monitors all dags and, as soon as their dependencies are fulfilled, schedules task instances to run.

While this process is going on in the background, the Airflow UI, served by the API server, displays information about the current dag and task statuses that it retrieves from the Airflow metadata database.

If you'd like to learn about Airflow components, you can check chapter 5 of [this practical guide](https://www.astronomer.io/ebooks/practical-guide-to-apache-airflow-3/?utm_source=deeplearning-ai&utm_medium=content&utm_campaign=genai-course-6-25).

### 3.3. Your First DAG

You'll now write your first day. The magic command `%%writefile` copies the content of the cell to the file `my_first_dag.py` stored under the `dags` folder. The `dags` folder, which is provided to you in this lab environment, will be automatically checked by the dag processor. Once the dag processor finds `my_first_dag.py`, it will automatically parse it and you can then view it in the Airflow UI.

#### 3.3.1. My first dag with 2 tasks

Run the following cell. After around 30 seconds, you should see the first dag in the UI. 

```python
%%writefile ../../dags/my_first_dag.py

from airflow.sdk import dag, task, chain


@dag
def my_first_dag():

    @task
    def my_task_1():
        return {"my_word" : "Airflow!"}
    
    _my_task_1 = my_task_1()

    @task 
    def my_task_2(my_dict):
        print(my_dict["my_word"])

    _my_task_2 = my_task_2(my_dict=_my_task_1)

    
my_first_dag()
```

**Note**: Where is the `dags` folder? In this environment, the `dags` folder lives at this address: `/home/jovyan/dags` (not in the lesson folders). You don't have direct access to the `dags` folder; but if you want to download all the dag files of this course, you can find them in this [github repo](https://github.com/astronomer/orchestrating-workflows-for-genai-deeplearning-ai). The repo also contains instructions on how to run Airflow locally.

#### 3.3.2. My first dag with 3 tasks

<div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>Airflow UI</code>:</b> 

<p>Changes to dags may take up to 30 seconds to show up in the Airflow UI in this environment! </p>
<p>In the Airflow UI, if you see the error "504 Gateway Timeout", this can happen after 2 hours or after some time of inactivity 25 minutes (if there's no activity for 20 minutes, the jupyter kernel stops and if there's no kernel for 5 minutes, then the jupyter notebook stops and the resources are released). In this case, make sure to refresh the notebook, run the cell that outputs the link to the Airflow UI and then use the link to open the Airflow UI. </p>
</div>

```python
%%writefile ../../dags/my_first_dag.py

from airflow.sdk import dag, task, chain


@dag
def my_first_dag():

    @task
    def my_task_1():
        return {"my_word" : "Airflow!"}
    
    _my_task_1 = my_task_1()

    @task 
    def my_task_2(my_dict):
        print(my_dict["my_word"])

    _my_task_2 = my_task_2(my_dict=_my_task_1)

    @task 
    def my_task_3():
        print("Hi from my_task_3!")

    _my_task_3 = my_task_3()

    chain(_my_task_1, _my_task_3)   
        

my_first_dag()
```

### 3.4. Your Second DAG

```python
%%writefile ../../dags/my_second_dag.py  
from airflow.sdk import dag, task, chain


@dag
def my_second_dag():
    @task
    def my_task_1():
        return 23

    _my_task_1 = my_task_1()

    @task
    def my_task_2():
        return 42

    _my_task_2 = my_task_2()

    @task
    def my_task_3(num1, num2):
        return num1 + num2

    _my_task_3 = my_task_3(num1=_my_task_1, num2=_my_task_2)

    @task
    def my_task_4():
        return "Math!"

    _my_task_4 = my_task_4()

    chain([_my_task_2, _my_task_3], _my_task_4)


my_second_dag()

```

### 3.5. Resources

How to install Airflow locally:

- If you're familiar with running Docker containers, you can check this guide: [Running Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
- If you'd like an easier approach to start with Airflow, you can use [Astro CLI](https://www.astronomer.io/docs/astro/cli/get-started-cli):
  - Make sure to check the last optional video of this course "How to Set up a Local Airflow Environment" that shows you how to replicate the same lab environment locally. It has this companion [github repo](https://github.com/astronomer/orchestrating-workflows-for-genai-deeplearning-ai).

Airflow features:  

- [Introduction to the TaskFlow API and Airflow decorators](https://www.astronomer.io/docs/learn/airflow-decorators/): Learn more about decorators generally in Python and specifically in Airflow.
- [Manage task and task group dependencies in Airflow](https://www.astronomer.io/docs/learn/managing-dependencies/): Learn more about setting dependencies between tasks using the `chain` function and other methods.
- [Airflow Operators](https://www.astronomer.io/docs/learn/what-is-an-operator): Learn more about operator classes which can be used alongside `@task` to create Airflow tasks.

<div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">

<p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>

<p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>

</div>

## Lesson 4: Turning Your Notebook Into a Pipeline

In this lesson, you will transform your RAG prototype of lesson 2 into two dags that you will manually trigger. The first dag will fetch the book descriptions, calculate the vector embeddings of the text descriptions and finally load the embeddings to a vector database. The second dag will query data from the vector database.

### 4.1. Link to Airflow UI

Run the following cell to get the link to the Airflow UI. If asked for username and password, make sure to type `airflow` for both.

```python
import os
airflow_ui = os.environ.get('DLAI_LOCAL_URL').format(port=8080)
airflow_ui #username:airflow password:airflow
```

You won't be using the two dags from the previous lesson (`my_first_dag` and `my_second_dag`). You will create two new dags `fetch_data` and `query_data`. Since the lab environment resets after 120 minutes, depending on when you're starting this lesson, you may not see the the dags of the previous lesson. If this is case, you don't need to worry about that because you will not use them in this lesson.

<div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>Airflow UI</code>:</b> 

<p>Changes to dags may take up to 30 seconds to show up in the Airflow UI in this environment! </p>
<p>In the Airflow UI, if you see the error "504 Gateway Timeout", this can happen after 2 hours or after some time of inactivity 25 minutes (if there's no activity for 20 minutes, the jupyter kernel stops and if there's no kernel for 5 minutes, then the jupyter notebook stops and the resources are released). In this case, make sure to refresh the notebook, run the cell that outputs the link to the Airflow UI and then use the link to open the Airflow UI. </p>
</div>

### 4.2. Creating two dags: fetch_data and query_data with empty tasks

Run the following two cells, and then check the UI after 30 seconds.

```python
%%writefile ../../dags/fetch_data.py 
from airflow.sdk import chain, dag, task 

@dag
def fetch_data():

    @task
    def create_collection_if_not_exists() -> None:
        pass

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        return []

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_files: list) -> list:
        return []

    _transform_book_description_files = transform_book_description_files(
        book_description_files=_list_book_description_files
    )

    @task
    def create_vector_embeddings(list_of_book_data: list) -> list:
        return []

    _create_vector_embeddings = create_vector_embeddings(
        list_of_book_data=_transform_book_description_files
    )

    @task
    def load_embeddings_to_vector_db(
        list_of_book_data: list, list_of_description_embeddings: list
    ) -> None:
        pass

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    chain(
        _create_collection_if_not_exists,
        _load_embeddings_to_vector_db
    )
    

fetch_data()
```

```python
%%writefile ../../dags/query_data.py 
from airflow.sdk import dag, task  


@dag
def query_data():

    @task
    def search_vector_db_for_a_book(query_str: str) -> None:
        pass

    search_vector_db_for_a_book(query_str="A philosophical book")


query_data()
```

### 4.3. Filling out the tasks of the two dags

**Optional reading notes related to the code in the next cell**: 

- `include/data`: in this and subsequent lessons, `include/data` is a directory that is provided to you (at `home/jovyan/include`) containing the text files of book descriptions. This directory is linked to the Airflow environment, so when the Airflow worker executes the tasks of reading from the text files, it knows where to find the data. Make sure to check the resource section below if you'd like to learn how to locally set up Airflow with this directory.
- How is Weaviate set up: it's set up as a local standalone server (if you're familiar with Docker, this [Docker image](https://github.com/astronomer/academy-genai/blob/main/docker-compose.override.yml) was used)
- What is `my_weaviate_conn`? 
  - Airflow has a Connection concept for storing credentials that are used to talk to external systems. A Connection is essentially set of parameters - such as username, password and hostname - along with the type of system that it connects to.
  - `my_weaviate_conn` represents the connection details that are needed to connect to weaviate.
  - you can enter these details manually in the Airflow UI before running the dag or you can define them as an environment variable when you set up Airflow. In this course, it's defined as this environment variable:
    ``` python
    AIRFLOW_CONN_MY_WEAVIATE_CONN='{
        "conn_type":"weaviate",
        "host":"localhost",
        "port":"8081",
        "extra":{
            "token":"adminkey",
            "additional_headers":{"X-Openai-Api-Key":"<YOUR OPENAI API KEY>"}, # not used in this course
            "grpc_port":"50051",
            "grpc_host":"localhost",
            "grpc_secure":"False",
            "http_secure":"False"
        }
    }'
    ```
    Make sure check the resource section below if you'd like to learn how to locally set up Airflow with Weaviate.
  - The connection details are passed to a `Weaviate hook`; a Hook is a high-level interface to an external platform that lets you quickly and easily talk to them without having to write low-level code that hits their API or uses special libraries. 

**Update the dags**:

Run the following two cells, and then check the UI after 30 seconds.

```python
%%writefile ../../dags/fetch_data.py 
from airflow.sdk import chain, dag, task 


COLLECTION_NAME = "Books" 
BOOK_DESCRIPTION_FOLDER = "/home/jovyan/include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dag
def fetch_data():

    @task
    def create_collection_if_not_exists() -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        existing_collections = client.collections.list_all()
        existing_collection_names = existing_collections.keys()

        if COLLECTION_NAME not in existing_collection_names:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            collection = client.collections.create(name=COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created successfully.")
            print(f"Collection details: {collection}")

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        import os
        
        book_description_files = [
            f for f in os.listdir(BOOK_DESCRIPTION_FOLDER)
            if f.endswith('.txt')
        ]
        return book_description_files

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_files: list) -> list:
        import json
        import os

        list_of_book_data = []
        
        for book_description_file in book_description_files:
            with open(
                os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
            ) as f:
                book_descriptions = f.readlines()
            
            titles = [
                book_description.split(":::")[1].strip()
                for book_description in book_descriptions
            ]
            authors = [
                book_description.split(":::")[2].strip()
                for book_description in book_descriptions
            ]
            book_description_text = [
                book_description.split(":::")[3].strip()
                for book_description in book_descriptions
            ]
            
            book_descriptions = [
                {
                    "title": title,
                    "author": author,
                    "description": description,
                }
                for title, author, description in zip(
                    titles, authors, book_description_text
                )
            ]
        
            list_of_book_data.append(book_descriptions)

        

        return list_of_book_data

    _transform_book_description_files = transform_book_description_files(
        book_description_files=_list_book_description_files
    )

    @task
    def create_vector_embeddings(list_of_book_data: list) -> list:
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)  
        
        list_of_description_embeddings = []
        
        for book_data in list_of_book_data:
            book_descriptions = [book["description"] for book in book_data]
            description_embeddings = [
                list(map(float, next(embedding_model.embed([desc])))) for desc in book_descriptions
            ]
        
            list_of_description_embeddings.append(description_embeddings)

        return list_of_description_embeddings

    _create_vector_embeddings = create_vector_embeddings(
        list_of_book_data=_transform_book_description_files
    )

    @task
    def load_embeddings_to_vector_db(
        list_of_book_data: list, list_of_description_embeddings: list
    ) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from weaviate.classes.data import DataObject

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()
        collection = client.collections.get(COLLECTION_NAME)

        for book_data_list, emb_list in zip(list_of_book_data, list_of_description_embeddings):
            items = []
            
            for book_data, emb in zip(book_data_list, emb_list):
                item = DataObject(
                    properties={
                        "title": book_data["title"],
                        "author": book_data["author"],
                        "description": book_data["description"],
                    },
                    vector=emb
                )
                items.append(item)
            
            collection.data.insert_many(items)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    chain(
        _create_collection_if_not_exists,
        _load_embeddings_to_vector_db
    )
    


fetch_data()
```

```python
%%writefile ../../dags/query_data.py 
from airflow.sdk import dag, task  

COLLECTION_NAME = "Books"  
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

@dag
def query_data():

    @task
    def search_vector_db_for_a_book(query_str: str) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from fastembed import TextEmbedding

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)  
        collection = client.collections.get(COLLECTION_NAME)
        
        query_emb = list(embedding_model.embed([query_str]))[0]
        
        results = collection.query.near_vector(
            near_vector=query_emb,
            limit=1,
        )
        for result in results.objects:
            print(f"You should read: {result.properties['title']} by {result.properties['author']}")
            print("Description:")
            print(result.properties["description"])

    search_vector_db_for_a_book(query_str="A philosophical book")


query_data()
```

### 4.4. Resources

- How to set up Airflow with weaviate locally using Astro CLI:
  - You can check the last optional video of this course "How to Set up a Local Airflow Environment" that shows you how to replicate the same lab environment locally. It has this companion [github repo](https://github.com/astronomer/orchestrating-workflows-for-genai-deeplearning-ai).
- [Connections & Hooks](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/connections.html#)
  
- [Airflow Weaviate Provider Package](https://airflow.apache.org/docs/apache-airflow-providers-weaviate/stable/index.html): Documentation of the Airflow Weaviate Provider Package which includes the `WeaviateHook`.
- [Airflow Hooks](https://www.astronomer.io/docs/learn/what-is-a-hook/): Learn about Airflow hooks like the `WeaviateHook`.
- [Manage connections in Apache Airflow](https://www.astronomer.io/docs/learn/connections): Learn about the different ways to connect Airflow to other tools.
- [Strategies for custom XCom backends in Airflow](https://www.astronomer.io/docs/learn/custom-xcom-backend-strategies/): Learn how to save data that is passed between tasks in different storage systems.

<div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">

<p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>

<p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>

</div>

## Lesson 5: Scheduling and Dag Parameters

In the previous lesson, you triggered the dags manually. In this lesson, you will learn how to schedule the fetch_data dag so that it automatically runs each hour (time-based scheduling), and how to schedule the query_data dag so that it runs after the embeddings are loaded to the database (data-aware scheduling).

### 5.1. Link to Airflow UI

Run the following cell to the link to the Airflow UI. If asked for username and password, make sure to type `airflow` for both.

```python
import os
airflow_ui = os.environ.get('DLAI_LOCAL_URL').format(port=8080)
airflow_ui #username:airflow password:airflow
```

<div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>Airflow UI</code>:</b> 

<p>Changes to dags may take up to 30 seconds to show up in the Airflow UI in this environment! </p>
<p>In the Airflow UI, if you see the error "504 Gateway Timeout", this can happen after 2 hours or after some time of inactivity 25 minutes (if there's no activity for 20 minutes, the jupyter kernel stops and if there's no kernel for 5 minutes, then the jupyter notebook stops and the resources are released). In this case, make sure to refresh the notebook, run the cell that outputs the link to the Airflow UI and then use the link to open the Airflow UI. </p>
</div>

### 5.2. Ensure Lesson 4 dags are in the Airflow UI

Since the lab environment resets after 120 minutes, depending on when you're starting this lesson, you may not see the the dags `fetch_data` and `query_data` from the previous lesson. To ensure the Airflow UI has the dags of the previous lesson, run the following two cells. After 30 seconds, check the Airflow UI. You should see the two dags, you can then manually trigger the dags. 

```python
%%writefile ../../dags/fetch_data.py 
from airflow.sdk import chain, dag, task 


COLLECTION_NAME = "Books" 
BOOK_DESCRIPTION_FOLDER = "/home/jovyan/include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dag
def fetch_data():

    @task
    def create_collection_if_not_exists() -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        existing_collections = client.collections.list_all()
        existing_collection_names = existing_collections.keys()

        if COLLECTION_NAME not in existing_collection_names:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            collection = client.collections.create(name=COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created successfully.")
            print(f"Collection details: {collection}")

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        import os
        
        book_description_files = [
            f for f in os.listdir(BOOK_DESCRIPTION_FOLDER)
            if f.endswith('.txt')
        ]
        return book_description_files

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_files: list) -> list:
        import json
        import os

        list_of_book_data = []
        
        for book_description_file in book_description_files:
            with open(
                os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
            ) as f:
                book_descriptions = f.readlines()
            
            titles = [
                book_description.split(":::")[1].strip()
                for book_description in book_descriptions
            ]
            authors = [
                book_description.split(":::")[2].strip()
                for book_description in book_descriptions
            ]
            book_description_text = [
                book_description.split(":::")[3].strip()
                for book_description in book_descriptions
            ]
            
            book_descriptions = [
                {
                    "title": title,
                    "author": author,
                    "description": description,
                }
                for title, author, description in zip(
                    titles, authors, book_description_text
                )
            ]
        
            list_of_book_data.append(book_descriptions)

        

        return list_of_book_data

    _transform_book_description_files = transform_book_description_files(
        book_description_files=_list_book_description_files
    )

    @task
    def create_vector_embeddings(list_of_book_data: list) -> list:
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)  
        
        list_of_description_embeddings = []
        
        for book_data in list_of_book_data:
            book_descriptions = [book["description"] for book in book_data]
            description_embeddings = [
                list(map(float, next(embedding_model.embed([desc])))) for desc in book_descriptions
            ]
        
            list_of_description_embeddings.append(description_embeddings)

        return list_of_description_embeddings

    _create_vector_embeddings = create_vector_embeddings(
        list_of_book_data=_transform_book_description_files
    )

    @task
    def load_embeddings_to_vector_db(
        list_of_book_data: list, list_of_description_embeddings: list
    ) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from weaviate.classes.data import DataObject

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()
        collection = client.collections.get(COLLECTION_NAME)

        for book_data_list, emb_list in zip(list_of_book_data, list_of_description_embeddings):
            items = []
            
            for book_data, emb in zip(book_data_list, emb_list):
                item = DataObject(
                    properties={
                        "title": book_data["title"],
                        "author": book_data["author"],
                        "description": book_data["description"],
                    },
                    vector=emb
                )
                items.append(item)
            
            collection.data.insert_many(items)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    chain(
        _create_collection_if_not_exists,
        _load_embeddings_to_vector_db
    )
    


fetch_data()
```

```python
%%writefile ../../dags/query_data.py 
from airflow.sdk import dag, task  

COLLECTION_NAME = "Books"  
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

@dag
def query_data():

    @task
    def search_vector_db_for_a_book(query_str: str) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from fastembed import TextEmbedding

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)  
        collection = client.collections.get(COLLECTION_NAME)
        
        query_emb = list(embedding_model.embed([query_str]))[0]
        
        results = collection.query.near_vector(
            near_vector=query_emb,
            limit=1,
        )
        for result in results.objects:
            print(f"You should read: {result.properties['title']} by {result.properties['author']}")
            print("Description:")
            print(result.properties["description"])

    search_vector_db_for_a_book(query_str="A philosophical book")


query_data()
```

### 5.3. Time-based scheduling: update fetch_data 

To make the dag run based on a schedule, you can update the `@dag decorator` by specifying the `start_date` and `schedule` parameters as shown in the following cell. The rest of the dag is the same for now.

Run the following cell to update the dag, and then check the Airflow UI. You don't need to trigger anything in the Airflow UI. 

```python
%%writefile ../../dags/fetch_data.py 
from airflow.sdk import chain, dag, task, Asset
from pendulum import datetime


COLLECTION_NAME = "Books"
BOOK_DESCRIPTION_FOLDER = "/home/jovyan/include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly"    
) #new!
def fetch_data():

    @task
    def create_collection_if_not_exists() -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        existing_collections = client.collections.list_all()
        existing_collection_names = existing_collections.keys()

        if COLLECTION_NAME not in existing_collection_names:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            collection = client.collections.create(name=COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created successfully.")
            print(f"Collection details: {collection}")

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        import os

        book_description_files = [
            f for f in os.listdir(BOOK_DESCRIPTION_FOLDER) if f.endswith(".txt")
        ]
        return book_description_files

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_files: list) -> list:
        import json
        import os

        list_of_book_data = []

        for book_description_file in book_description_files:
            with open(
                os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
            ) as f:
                book_descriptions = f.readlines()

            titles = [
                book_description.split(":::")[1].strip()
                for book_description in book_descriptions
            ]
            authors = [
                book_description.split(":::")[2].strip()
                for book_description in book_descriptions
            ]
            book_description_text = [
                book_description.split(":::")[3].strip()
                for book_description in book_descriptions
            ]

            book_descriptions = [
                {
                    "title": title,
                    "author": author,
                    "description": description,
                }
                for title, author, description in zip(
                    titles, authors, book_description_text
                )
            ]

            list_of_book_data.append(book_descriptions)

        return list_of_book_data

    _transform_book_description_files = transform_book_description_files(
        book_description_files=_list_book_description_files
    )

    @task
    def create_vector_embeddings(list_of_book_data: list) -> list:
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)

        list_of_description_embeddings = []

        for book_data in list_of_book_data:
            book_descriptions = [book["description"] for book in book_data]
            description_embeddings = [
                list(map(float, next(embedding_model.embed([desc]))))
                for desc in book_descriptions
            ]

            list_of_description_embeddings.append(description_embeddings)

        return list_of_description_embeddings

    _create_vector_embeddings = create_vector_embeddings(
        list_of_book_data=_transform_book_description_files
    )

    @task
    def load_embeddings_to_vector_db(
        list_of_book_data: list, list_of_description_embeddings: list
    ) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from weaviate.classes.data import DataObject

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()
        collection = client.collections.get(COLLECTION_NAME)

        for book_data_list, emb_list in zip(
            list_of_book_data, list_of_description_embeddings
        ):
            items = []

            for book_data, emb in zip(book_data_list, emb_list):
                item = DataObject(
                    properties={
                        "title": book_data["title"],
                        "author": book_data["author"],
                        "description": book_data["description"],
                    },
                    vector=emb,
                )
                items.append(item)

            collection.data.insert_many(items)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    chain(_create_collection_if_not_exists, _load_embeddings_to_vector_db)


fetch_data()
```

### 5.4. Data-aware scheduling for query_data 

Here are the updates you will make to both dags to make query_data data aware.

<img src="updates2.png" width="400">

1. You will first update the parameter of the last task `load_embeddings_to_vector_db` of the fetch_data dag: 
     ```
   @task(
        outlets=[Asset("my_book_vector_data")]
    )
    ```
   This means that when the task is done it will emit an AssetEvent informing the Asset object that the collection books has been updated.


2. You will then update the parameter of the dag query_data:
   ```
   @dag(
    schedule=[Asset("my_book_vector_data")]
    )
   ``` 
   This means that the second dag will be triggered whenever there is an update to the collection books in the weaviate database.

**5.4.1. Update the last task of the dag: fetch_data**

Run the following cell and then check the Airflow UI. 

```python
%%writefile ../../dags/fetch_data.py 
from airflow.sdk import chain, dag, task, Asset
from pendulum import datetime


COLLECTION_NAME = "Books"
BOOK_DESCRIPTION_FOLDER = "/home/jovyan/include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly"    
)
def fetch_data():

    @task
    def create_collection_if_not_exists() -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        existing_collections = client.collections.list_all()
        existing_collection_names = existing_collections.keys()

        if COLLECTION_NAME not in existing_collection_names:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            collection = client.collections.create(name=COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created successfully.")
            print(f"Collection details: {collection}")

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        import os

        book_description_files = [
            f for f in os.listdir(BOOK_DESCRIPTION_FOLDER) if f.endswith(".txt")
        ]
        return book_description_files

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_files: list) -> list:
        import json
        import os

        list_of_book_data = []

        for book_description_file in book_description_files:
            with open(
                os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
            ) as f:
                book_descriptions = f.readlines()

            titles = [
                book_description.split(":::")[1].strip()
                for book_description in book_descriptions
            ]
            authors = [
                book_description.split(":::")[2].strip()
                for book_description in book_descriptions
            ]
            book_description_text = [
                book_description.split(":::")[3].strip()
                for book_description in book_descriptions
            ]

            book_descriptions = [
                {
                    "title": title,
                    "author": author,
                    "description": description,
                }
                for title, author, description in zip(
                    titles, authors, book_description_text
                )
            ]

            list_of_book_data.append(book_descriptions)

        return list_of_book_data

    _transform_book_description_files = transform_book_description_files(
        book_description_files=_list_book_description_files
    )

    @task
    def create_vector_embeddings(list_of_book_data: list) -> list:
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)

        list_of_description_embeddings = []

        for book_data in list_of_book_data:
            book_descriptions = [book["description"] for book in book_data]
            description_embeddings = [
                list(map(float, next(embedding_model.embed([desc]))))
                for desc in book_descriptions
            ]

            list_of_description_embeddings.append(description_embeddings)

        return list_of_description_embeddings

    _create_vector_embeddings = create_vector_embeddings(
        list_of_book_data=_transform_book_description_files
    )

    @task(
        outlets=[Asset("my_book_vector_data")]
    ) #new!
    def load_embeddings_to_vector_db(
        list_of_book_data: list, list_of_description_embeddings: list
    ) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from weaviate.classes.data import DataObject

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()
        collection = client.collections.get(COLLECTION_NAME)

        for book_data_list, emb_list in zip(
            list_of_book_data, list_of_description_embeddings
        ):
            items = []

            for book_data, emb in zip(book_data_list, emb_list):
                item = DataObject(
                    properties={
                        "title": book_data["title"],
                        "author": book_data["author"],
                        "description": book_data["description"],
                    },
                    vector=emb,
                )
                items.append(item)

            collection.data.insert_many(items)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    chain(_create_collection_if_not_exists, _load_embeddings_to_vector_db)


fetch_data()
```

**5.4.2. Update query_data**

Run the following cell and check the Airflow UI.

**Note**: If you try the same query of the video, you might get a different book recommendation (in the video, some additional book descriptions were given to the embedding model when creating the vector database). 

```python
%%writefile ../../dags/query_data.py

from airflow.sdk import dag, task, Asset

COLLECTION_NAME = "Books"  
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dag(
    schedule=[Asset("my_book_vector_data")]
) #new!
def query_data():

    @task
    def search_vector_db_for_a_book(query_str: str):
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from fastembed import TextEmbedding

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)  
        collection = client.collections.get(COLLECTION_NAME)
        
        query_emb = list(embedding_model.embed([query_str]))[0]
        
        results = collection.query.near_vector(
            near_vector=query_emb,
            limit=1,
        )
        for result in results.objects:
            print(f"You should read: {result.properties['title']} by {result.properties['author']}")
            print("Description:")
            print(result.properties["description"])

    search_vector_db_for_a_book(query_str="A philosophical book")


query_data()
```

### 5.5. Adding `params` parameter to the query_data dag

Instead of hardcoding the query in the dag, you can now allow the user to specify the query by specific the `params` for the dag decorator. Run the following cell, it might take around 30 seconds for the dag to be updated in the Airflow UI.

```python
%%writefile ../../dags/query_data.py

from airflow.sdk import dag, task, Asset

COLLECTION_NAME = "Books"  
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dag(
    schedule=[Asset("my_book_vector_data")],
    params={
        "query_str":"A philosophical book"
    }
) #new!
def query_data():

    @task
    def search_vector_db_for_a_book(**context):
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from fastembed import TextEmbedding

        query_str=context["params"]["query_str"] #new!

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)  
        collection = client.collections.get(COLLECTION_NAME)
        
        query_emb = list(embedding_model.embed([query_str]))[0]
        
        results = collection.query.near_vector(
            near_vector=query_emb,
            limit=1,
        )
        for result in results.objects:
            print(f"You should read: {result.properties['title']} by {result.properties['author']}")
            print("Description:")
            print(result.properties["description"])

    search_vector_db_for_a_book()


query_data()
```

**Optional Part**: Add your own book files

Feel free to add your own book description and then trigger the fetch_data dag.

```python
# Add your own book description file
# Format 
# [Integer Index] ::: [Book Title] ([Release year]) ::: [Author] ::: [Description]

my_book_description = """0 ::: The Idea of the World (2019) ::: Bernardo Kastrup ::: An ontological thesis arguing for the primacy of mind over matter.
1 ::: Exploring the World of Lucid Dreaming (1990) ::: Stephen LaBerge ::: A practical guide to learning and enjoying lucid dreams.
"""

my_book_description_file_name = "my_descs_1.txt"

# Write to file
with open(f"../../include/data/{my_book_description_file_name}", 'w') as f:
    f.write(my_book_description)
```

```python
# ## Remove a book description file
 
# import os

# my_book_description_file_name = "my_descs_1.txt"

# file_path = f"../../include/data/{my_book_description_file_name}"

# # Remove the file
# if os.path.exists(file_path):
#     os.remove(file_path)
# else:
#     print(f"File not found: {file_path}")
```

### 5.6. Resources

- [Schedule DAGs in Apache Airflow®](https://www.astronomer.io/docs/learn/scheduling-in-airflow/): Learn all the different ways of scheduling Airflow dags.
- [DAG-level parameters in Airflow](https://www.astronomer.io/docs/learn/airflow-dag-parameters/): A comprehensive list of dag parameters in Airflow.
- [Assets and data-aware scheduling in Airflow](https://www.astronomer.io/docs/learn/airflow-datasets/): Learn how to created advanced data-aware schedules using `Asset`s in Airflow.
- [Access the Apache Airflow context](https://www.astronomer.io/docs/learn/airflow-context/): Learn how to interact with the Airflow context dictionary retrieved with `**context`.
- [Create and use params in Airflow](https://www.astronomer.io/docs/learn/airflow-params/): Learn how to create advanced `params` dictionaries for your Airflow dags.
- [Airflow REST API - Create Asset Event](https://airflow.apache.org/docs/apache-airflow/stable/stable-rest-api-ref.html#operation/create_asset_event): You can update Assets from outside of Airflow using the Airflow REST API.

<div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">

<p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>

<p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>

</div>

# Lesson 6: Make the Pipeline Adaptable

You now use the concept of dynamic task mapping so that the book information is extracted from each text file in parallel.

### 6.1. Link to Airflow UI

Run the following cell to the link to the Airflow UI. If asked for username and password, make sure to type `airflow` for both.

```python
import os
airflow_ui = os.environ.get('DLAI_LOCAL_URL').format(port=8080)
airflow_ui #username:airflow password:airflow
```

<div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>Airflow UI</code>:</b> 

<p>Changes to dags may take up to 30 seconds to show up in the Airflow UI in this environment! </p>
<p>In the Airflow UI, if you see the error "504 Gateway Timeout", this can happen after 2 hours or after some time of inactivity 25 minutes (if there's no activity for 20 minutes, the jupyter kernel stops and if there's no kernel for 5 minutes, then the jupyter notebook stops and the resources are released). In this case, make sure to refresh the notebook, run the cell that outputs the link to the Airflow UI and then use the link to open the Airflow UI. </p>
</div>

### 6.2. What is the issue with the current fetch_data dag

**Note:** The video starts with showing you the code from the previous lesson to explain the current issue with the dag: `fetch_data`

In the task `transform_book_description_files`, you're iterating through all of the text files in one task and repeat the same process. If the task fails due to a formatting error in one text file, then you would need to repeat again the same processing for all the files. The better approach is to process the text files in parallel, i.e., transform the task `transform_book_description_files` into parallel tasks, where each task processes one text file. You'll learn how to do that using Dynamic Task Mapping.

```python
@task
    def transform_book_description_files(book_description_files: list) -> list:
        import json
        import os

        list_of_book_data = []
        
        for book_description_file in book_description_files:
            with open(
                os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
            ) as f:
                book_descriptions = f.readlines()
            
            # the rest of the code
            # .....
        
            list_of_book_data.append(book_descriptions)

        

        return list_of_book_data

```

### 6.3. Simple dynamic task mapping example

Here's a dag showing a simple dynamic task mapping example. Make sure to run the cell and trigger the dag in the Airflow UI.

```python
%%writefile ../../dags/simple_dynamic_task_mapping.py 

from airflow.sdk import dag, task 


@dag
def simple_mapping():

    @task
    def get_numbers():
        import random
        
        return [_ for _ in range(random.randint(0, 3))]  # [0,1,2]

    _get_numbers = get_numbers()


    @task
    def mapped_task_one(my_constant_arg: int, my_changing_arg: int):
        return my_constant_arg + my_changing_arg

    _mapped_task_one = mapped_task_one.partial(
        my_constant_arg=10
    ).expand(my_changing_arg=_get_numbers)
  

simple_mapping()
```

Adding another task to the same simple example. Make sure to run the cell and trigger the dag in the Airflow UI.

```python
%%writefile ../../dags/simple_dynamic_task_mapping.py 

from airflow.sdk import dag, task 


@dag
def simple_mapping():

    @task
    def get_numbers():
        import random
        
        return [_ for _ in range(random.randint(0, 3))]  # [0,1,2]

    _get_numbers = get_numbers()


    @task
    def mapped_task_one(my_constant_arg: int, my_changing_arg: int):
        return my_constant_arg + my_changing_arg

    _mapped_task_one = mapped_task_one.partial(
        my_constant_arg=10
    ).expand(my_changing_arg=_get_numbers)


    @task
    def mapped_task_two(my_cookie_number: int):
        print(f"There are {my_cookie_number} cookies in the jar!")

    mapped_task_two.expand(my_cookie_number=_mapped_task_one)    

simple_mapping()
```

### 6.4. Applying dynamic task mapping to `fetch_data`

Dynamically map the `transform_book_description_files` task over `list_book_description_files`.

Here are the changes made to `transform_book_description_files`:
- the input was changed from list to string
- the outer for-loop was removed (that looped through the text files)
- the descriptions of the books in one file `book_descriptions` are returned
- the `.expand` was used to change the task into a dynamic task

```python
    _transform_book_description_files = transform_book_description_files.expand(
        book_description_file=_list_book_description_files
    )
```


Here are the changes made to `create_vector_embeddings`:
- the input was changed to `book_data`
- the outer for-loop was removed
- the description embeddings are directly returned
- the .expand was used to change the task into a dynamic task

```python
    _create_vector_embeddings = create_vector_embeddings.expand(
        book_data=_transform_book_description_files
    )
```

Run the following cell, wait for 30 seconds then check the Airflow UI to trigger the dag. 

**Note:** Depending on when you're starting this lesson, if the Airflow UI does not have the dags from the previous lesson and this is the first time you write the dag to the folder `dags`, you can just unpause the dag. It will run automatically since it's scheduled to run every hour. If you trigger the dag, you will see two runs: the automatic one and the triggered one. 

```python
%%writefile ../../dags/fetch_data.py 

from airflow.sdk import chain, dag, task, Asset
from pendulum import datetime

COLLECTION_NAME = "Books"
BOOK_DESCRIPTION_FOLDER = "/home/jovyan/include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly"
)
def fetch_data():

    @task
    def create_collection_if_not_exists() -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        existing_collections = client.collections.list_all()
        existing_collection_names = existing_collections.keys()

        if COLLECTION_NAME not in existing_collection_names:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            collection = client.collections.create(name=COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created successfully.")
            print(f"Collection details: {collection}")

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        import os

        book_description_files = [
            f for f in os.listdir(BOOK_DESCRIPTION_FOLDER) if f.endswith(".txt")
        ]
        return book_description_files

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_file: str) -> str:
        import json
        import os

        with open(
            os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
        ) as f:
            book_descriptions = f.readlines()

        titles = [
            book_description.split(":::")[1].strip()
            for book_description in book_descriptions
        ]
        authors = [
            book_description.split(":::")[2].strip()
            for book_description in book_descriptions
        ]
        book_description_text = [
            book_description.split(":::")[3].strip()
            for book_description in book_descriptions
        ]

        book_descriptions = [
            {
                "title": title,
                "author": author,
                "description": description,
            }
            for title, author, description in zip(
                titles, authors, book_description_text
            )
        ]

        return book_descriptions

    _transform_book_description_files = transform_book_description_files.expand(
        book_description_file=_list_book_description_files
    )

    @task
    def create_vector_embeddings(book_data: list) -> list:
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)


        book_descriptions = [book["description"] for book in book_data]
        description_embeddings = [
            list(map(float, next(embedding_model.embed([desc]))))
            for desc in book_descriptions
        ]



        return description_embeddings

    _create_vector_embeddings = create_vector_embeddings.expand(
        book_data=_transform_book_description_files
    )

    @task(
        outlets=[Asset("my_book_vector_data")]
    )
    def load_embeddings_to_vector_db(
        list_of_book_data: list, list_of_description_embeddings: list
    ) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from weaviate.classes.data import DataObject

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()
        collection = client.collections.get(COLLECTION_NAME)

        for book_data_list, emb_list in zip(
            list_of_book_data, list_of_description_embeddings
        ):
            items = []

            for book_data, emb in zip(book_data_list, emb_list):
                item = DataObject(
                    properties={
                        "title": book_data["title"],
                        "author": book_data["author"],
                        "description": book_data["description"],
                    },
                    vector=emb,
                )
                items.append(item)

            collection.data.insert_many(items)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    chain(_create_collection_if_not_exists, _load_embeddings_to_vector_db)


fetch_data()
```

**Optional Part**: Add your own book files

Feel free to add your own book description and then trigger the fetch_data dag.

```python
# Add your own book description file
# Format 
# [Integer Index] ::: [Book Title] ([Release year]) ::: [Author] ::: [Description]

my_book_description = """0 ::: The Idea of the World (2019) ::: Bernardo Kastrup ::: An ontological thesis arguing for the primacy of mind over matter.
1 ::: Exploring the World of Lucid Dreaming (1990) ::: Stephen LaBerge ::: A practical guide to learning and enjoying lucid dreams.
"""

my_book_description_file_name = "my_descs_1.txt"

# Write to file
with open(f"../../include/data/{my_book_description_file_name}", 'w') as f:
    f.write(my_book_description)
```

```python
# ## Remove a book description file
 
# import os

# my_book_description_file_name = "my_descs_1.txt"

# file_path = f"../../include/data/{my_book_description_file_name}"

# # Remove the file
# if os.path.exists(file_path):
#     os.remove(file_path)
# else:
#     print(f"File not found: {file_path}")
```

## 6.5. Resources

- [Create dynamic Airflow tasks](https://www.astronomer.io/docs/learn/dynamic-tasks/): Learn all about dynamic task mapping in Airflow.
- Tip: you can limit the number of concurrently running mapped task instances using the task-level parameters `max_active_tis_per_dag` and `max_active_tis_per_dagrun`.
- [Airflow configuration reference - AIRFLOW__CORE__MAX_MAP_LENGTH](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html#max-map-length): By default you can have up to 1024 dynamically mapped instances per task. Use this configuration environment variable to modify that limit.

<div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">

<p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>

<p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>

</div>

## Lesson 7: Prepare to Fail

In this lesson, you will learn how to configure retries and callback features in case of dag and task failures.


### 7.1. Link to Airflow UI

Run the following cell to the link to the Airflow UI. If asked for username and password, make sure to type `airflow` for both.

```python
import os
airflow_ui = os.environ.get('DLAI_LOCAL_URL').format(port=8080)
airflow_ui #username:airflow password:airflow
```

<div style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>Airflow UI</code>:</b> 

<p>Changes to dags may take up to 30 seconds to show up in the Airflow UI in this environment! </p>
<p>In the Airflow UI, if you see the error "504 Gateway Timeout", this can happen after 2 hours or after some time of inactivity 25 minutes (if there's no activity for 20 minutes, the jupyter kernel stops and if there's no kernel for 5 minutes, then the jupyter notebook stops and the resources are released). In this case, make sure to refresh the notebook, run the cell that outputs the link to the Airflow UI and then use the link to open the Airflow UI. </p>
</div>

### 7.2. Exercise: Make a task fail

The line `print(10/0)` has been added to the first task. 
1. Run the cell, wait for 30 seconds and then trigger the dag in the Airflow UI. 
2. Remove the line `print(10/0)` , run the cell again, wait for 15-30 seconds and then retry the task instance by clicking its Clear button in the Airflow UI.


**Note:** Depending on when you're starting this lesson, if the Airflow UI does not have the dags from the previous lesson and this is the first time you write the dag to the folder `dags`, you can just unpause the dag. It will run automatically since it's scheduled to run every hour. If you trigger the dag, you will see two runs: the automatic one and the triggered one. 

```python
%%writefile ../../dags/fetch_data.py 

from airflow.sdk import chain, dag, task, Asset
from pendulum import datetime

COLLECTION_NAME = "Books"
BOOK_DESCRIPTION_FOLDER = "/home/jovyan/include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly"
)
def fetch_data():

    @task
    def create_collection_if_not_exists() -> None:
        print(10/0)
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        existing_collections = client.collections.list_all()
        existing_collection_names = existing_collections.keys()

        if COLLECTION_NAME not in existing_collection_names:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            collection = client.collections.create(name=COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created successfully.")
            print(f"Collection details: {collection}")

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        import os

        book_description_files = [
            f for f in os.listdir(BOOK_DESCRIPTION_FOLDER) if f.endswith(".txt")
        ]
        return book_description_files

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_file: str) -> str:
        import json
        import os

        with open(
            os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
        ) as f:
            book_descriptions = f.readlines()

        titles = [
            book_description.split(":::")[1].strip()
            for book_description in book_descriptions
        ]
        authors = [
            book_description.split(":::")[2].strip()
            for book_description in book_descriptions
        ]
        book_description_text = [
            book_description.split(":::")[3].strip()
            for book_description in book_descriptions
        ]

        book_descriptions = [
            {
                "title": title,
                "author": author,
                "description": description,
            }
            for title, author, description in zip(
                titles, authors, book_description_text
            )
        ]

        return book_descriptions

    _transform_book_description_files = transform_book_description_files.expand(
        book_description_file=_list_book_description_files
    )

    @task
    def create_vector_embeddings(book_data: list) -> list:
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)


        book_descriptions = [book["description"] for book in book_data]
        description_embeddings = [
            list(map(float, next(embedding_model.embed([desc]))))
            for desc in book_descriptions
        ]



        return description_embeddings

    _create_vector_embeddings = create_vector_embeddings.expand(
        book_data=_transform_book_description_files
    )

    @task(
        outlets=[Asset("my_book_vector_data")]
    )
    def load_embeddings_to_vector_db(
        list_of_book_data: list, list_of_description_embeddings: list
    ) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from weaviate.classes.data import DataObject

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()
        collection = client.collections.get(COLLECTION_NAME)

        for book_data_list, emb_list in zip(
            list_of_book_data, list_of_description_embeddings
        ):
            items = []

            for book_data, emb in zip(book_data_list, emb_list):
                item = DataObject(
                    properties={
                        "title": book_data["title"],
                        "author": book_data["author"],
                        "description": book_data["description"],
                    },
                    vector=emb,
                )
                items.append(item)

            collection.data.insert_many(items)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    chain(_create_collection_if_not_exists, _load_embeddings_to_vector_db)


fetch_data()
```

### 7.3. Add retries to the dag

To address transient failures, you can configure retries at the dag level.

```python
@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly",
    default_args={
        "retries": 1,
        "retry_delay": duration(seconds=10)
    }
)
```
To check the `retries` behavior in the Airflow UI, run the following cell and then check the Airflow UI. Make sure to trigger the dag.


```python
%%writefile ../../dags/fetch_data.py 

from airflow.sdk import chain, dag, task, Asset
from pendulum import datetime, duration

COLLECTION_NAME = "Books"
BOOK_DESCRIPTION_FOLDER = "/home/jovyan/include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly",
    default_args={
        "retries": 1,
        "retry_delay": duration(seconds=10)
    }
)#new!
def fetch_data():

    @task
    def create_collection_if_not_exists() -> None:
        print(10/0)
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        existing_collections = client.collections.list_all()
        existing_collection_names = existing_collections.keys()

        if COLLECTION_NAME not in existing_collection_names:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            collection = client.collections.create(name=COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created successfully.")
            print(f"Collection details: {collection}")

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        import os

        book_description_files = [
            f for f in os.listdir(BOOK_DESCRIPTION_FOLDER) if f.endswith(".txt")
        ]
        return book_description_files

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_file: str) -> str:
        import json
        import os

        with open(
            os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
        ) as f:
            book_descriptions = f.readlines()

        titles = [
            book_description.split(":::")[1].strip()
            for book_description in book_descriptions
        ]
        authors = [
            book_description.split(":::")[2].strip()
            for book_description in book_descriptions
        ]
        book_description_text = [
            book_description.split(":::")[3].strip()
            for book_description in book_descriptions
        ]

        book_descriptions = [
            {
                "title": title,
                "author": author,
                "description": description,
            }
            for title, author, description in zip(
                titles, authors, book_description_text
            )
        ]

        return book_descriptions

    _transform_book_description_files = transform_book_description_files.expand(
        book_description_file=_list_book_description_files
    )

    @task
    def create_vector_embeddings(book_data: list) -> list:
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)


        book_descriptions = [book["description"] for book in book_data]
        description_embeddings = [
            list(map(float, next(embedding_model.embed([desc]))))
            for desc in book_descriptions
        ]



        return description_embeddings

    _create_vector_embeddings = create_vector_embeddings.expand(
        book_data=_transform_book_description_files
    )

    @task(
        outlets=[Asset("my_book_vector_data")]
    )
    def load_embeddings_to_vector_db(
        list_of_book_data: list, list_of_description_embeddings: list
    ) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from weaviate.classes.data import DataObject

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()
        collection = client.collections.get(COLLECTION_NAME)

        for book_data_list, emb_list in zip(
            list_of_book_data, list_of_description_embeddings
        ):
            items = []

            for book_data, emb in zip(book_data_list, emb_list):
                item = DataObject(
                    properties={
                        "title": book_data["title"],
                        "author": book_data["author"],
                        "description": book_data["description"],
                    },
                    vector=emb,
                )
                items.append(item)

            collection.data.insert_many(items)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    chain(_create_collection_if_not_exists, _load_embeddings_to_vector_db)


fetch_data()
```

### 7.4. Add retries to the first task & Configure the trigger rule for the last task

```python
%%writefile ../../dags/fetch_data.py 

from airflow.sdk import chain, dag, task, Asset
from pendulum import datetime, duration

COLLECTION_NAME = "Books"
BOOK_DESCRIPTION_FOLDER = "/home/jovyan/include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly",
    default_args={
        "retries": 1,
        "retry_delay": duration(seconds=10)
    }
)
def fetch_data():

    @task(retries=5, retry_delay=duration(seconds=2))
    def create_collection_if_not_exists() -> None:
        print(10/0)
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        existing_collections = client.collections.list_all()
        existing_collection_names = existing_collections.keys()

        if COLLECTION_NAME not in existing_collection_names:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            collection = client.collections.create(name=COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created successfully.")
            print(f"Collection details: {collection}")

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        import os

        book_description_files = [
            f for f in os.listdir(BOOK_DESCRIPTION_FOLDER) if f.endswith(".txt")
        ]
        return book_description_files

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_file: str) -> str:
        import json
        import os

        with open(
            os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
        ) as f:
            book_descriptions = f.readlines()

        titles = [
            book_description.split(":::")[1].strip()
            for book_description in book_descriptions
        ]
        authors = [
            book_description.split(":::")[2].strip()
            for book_description in book_descriptions
        ]
        book_description_text = [
            book_description.split(":::")[3].strip()
            for book_description in book_descriptions
        ]

        book_descriptions = [
            {
                "title": title,
                "author": author,
                "description": description,
            }
            for title, author, description in zip(
                titles, authors, book_description_text
            )
        ]

        return book_descriptions

    _transform_book_description_files = transform_book_description_files.expand(
        book_description_file=_list_book_description_files
    )

    @task
    def create_vector_embeddings(book_data: list) -> list:
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)


        book_descriptions = [book["description"] for book in book_data]
        description_embeddings = [
            list(map(float, next(embedding_model.embed([desc]))))
            for desc in book_descriptions
        ]



        return description_embeddings

    _create_vector_embeddings = create_vector_embeddings.expand(
        book_data=_transform_book_description_files
    )

    @task(
        outlets=[Asset("my_book_vector_data")],
        trigger_rule="all_done" # new!
    )
    def load_embeddings_to_vector_db(
        list_of_book_data: list, list_of_description_embeddings: list
    ) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from weaviate.classes.data import DataObject

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()
        collection = client.collections.get(COLLECTION_NAME)

        for book_data_list, emb_list in zip(
            list_of_book_data, list_of_description_embeddings
        ):
            items = []

            for book_data, emb in zip(book_data_list, emb_list):
                item = DataObject(
                    properties={
                        "title": book_data["title"],
                        "author": book_data["author"],
                        "description": book_data["description"],
                    },
                    vector=emb,
                )
                items.append(item)

            collection.data.insert_many(items)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    chain(_create_collection_if_not_exists, _load_embeddings_to_vector_db)


fetch_data()
```

### 7.5. Add an on_failure_callback to the dag

```python
%%writefile ../../dags/fetch_data.py 

from airflow.sdk import chain, dag, task, Asset
from pendulum import datetime, duration

COLLECTION_NAME = "Books"
BOOK_DESCRIPTION_FOLDER = "/home/jovyan/include/data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def _my_callback_func(context):
    task_instance = context["task_instance"]
    dag_run = context["dag_run"]
    print(
        f"CALLBACK: Task {task_instance.task_id} "
        f"failed in DAG {dag_run.dag_id} at {dag_run.start_date}"
    )

@dag(
    start_date=datetime(2025, 4, 1),
    schedule="@hourly",
    default_args={
        "retries": 1,
        "retry_delay": duration(seconds=10),
        "on_failure_callback": _my_callback_func, 
    },
    on_failure_callback=_my_callback_func
)
def fetch_data():

    @task(retries=5, retry_delay=duration(seconds=2))
    def create_collection_if_not_exists() -> None:
        print(10/0)
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()

        existing_collections = client.collections.list_all()
        existing_collection_names = existing_collections.keys()

        if COLLECTION_NAME not in existing_collection_names:
            print(f"Collection {COLLECTION_NAME} does not exist yet. Creating it...")
            collection = client.collections.create(name=COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created successfully.")
            print(f"Collection details: {collection}")

    _create_collection_if_not_exists = create_collection_if_not_exists()

    @task
    def list_book_description_files() -> list:
        import os

        book_description_files = [
            f for f in os.listdir(BOOK_DESCRIPTION_FOLDER) if f.endswith(".txt")
        ]
        return book_description_files

    _list_book_description_files = list_book_description_files()

    @task
    def transform_book_description_files(book_description_file: str) -> str:
        import json
        import os

        with open(
            os.path.join(BOOK_DESCRIPTION_FOLDER, book_description_file), "r"
        ) as f:
            book_descriptions = f.readlines()

        titles = [
            book_description.split(":::")[1].strip()
            for book_description in book_descriptions
        ]
        authors = [
            book_description.split(":::")[2].strip()
            for book_description in book_descriptions
        ]
        book_description_text = [
            book_description.split(":::")[3].strip()
            for book_description in book_descriptions
        ]

        book_descriptions = [
            {
                "title": title,
                "author": author,
                "description": description,
            }
            for title, author, description in zip(
                titles, authors, book_description_text
            )
        ]

        return book_descriptions

    _transform_book_description_files = transform_book_description_files.expand(
        book_description_file=_list_book_description_files
    )

    @task
    def create_vector_embeddings(book_data: list) -> list:
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)


        book_descriptions = [book["description"] for book in book_data]
        description_embeddings = [
            list(map(float, next(embedding_model.embed([desc]))))
            for desc in book_descriptions
        ]



        return description_embeddings

    _create_vector_embeddings = create_vector_embeddings.expand(
        book_data=_transform_book_description_files
    )

    @task(
        outlets=[Asset("my_book_vector_data")],
        trigger_rule="all_done" # new!
    )
    def load_embeddings_to_vector_db(
        list_of_book_data: list, list_of_description_embeddings: list
    ) -> None:
        from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
        from weaviate.classes.data import DataObject

        hook = WeaviateHook("my_weaviate_conn")
        client = hook.get_conn()
        collection = client.collections.get(COLLECTION_NAME)

        for book_data_list, emb_list in zip(
            list_of_book_data, list_of_description_embeddings
        ):
            items = []

            for book_data, emb in zip(book_data_list, emb_list):
                item = DataObject(
                    properties={
                        "title": book_data["title"],
                        "author": book_data["author"],
                        "description": book_data["description"],
                    },
                    vector=emb,
                )
                items.append(item)

            collection.data.insert_many(items)

    _load_embeddings_to_vector_db = load_embeddings_to_vector_db(
        list_of_book_data=_transform_book_description_files,
        list_of_description_embeddings=_create_vector_embeddings,
    )

    chain(_create_collection_if_not_exists, _load_embeddings_to_vector_db)


fetch_data()
```

### 7.6. Resources

- [Airflow trigger rules](https://www.astronomer.io/docs/learn/airflow-trigger-rules/): A reference of all available trigger rules.
- [Manage Apache Airflow® DAG notifications](https://www.astronomer.io/docs/learn/error-notifications-in-airflow/): Learn about different ways to let Airflow notify you of task and dag states, including notifier classes.
- [Airflow Apprise provider](https://airflow.apache.org/docs/apache-airflow-providers-apprise/stable/index.html): Documentation for the Airflow Apprise provider that integrates with many notification tools.
- Deploy Airflow pipelines to the cloud using a [free trial of Astro](https://www.astronomer.io/lp/signup/?utm_source=deeplearning-ai&utm_medium=content&utm_campaign=genai-course-6-25) 

<div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">

<p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>

<p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>

</div>