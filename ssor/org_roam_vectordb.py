from config import persist_directory
from org_roam_parser import org_roam_df
from langchain.docstore.document import Document

from langchain.vectorstores import Chroma

from langchain.embeddings import HuggingFaceInstructEmbeddings


def org_roam_vectordb():
    embedding = HuggingFaceInstructEmbeddings(
        query_instruction="Represent the query for retrieval: "
    )

    vectordb = Chroma(
        "langchain_store",
        embedding_function=embedding,
        persist_directory=persist_directory,
    )

    roam_df = org_roam_df()
    text_result = []
    metadata_result = []
    ids_result = []
    for index, row in roam_df.iterrows():
        hash = row["node_hash"]
        if not vectordb.get(where={"hash": hash})["ids"]:
            org_id = row["node_id"]
            title = row["node_title"]
            file_name = row["file_name"]
            node_hierarchy = row["node_hierarchy"]
            texts = row["text_to_encode"].split("\n\n\n")

            texts = ["[" + node_hierarchy + "] " + text for text in texts]
            metadatas = [
                {
                    "source": f"{index}-{i}",
                    "ID": org_id,
                    "title": title,
                    "hierarchy": node_hierarchy,
                    "file_name": file_name,
                    "hash": hash,
                    "body": texts[i],
                }
                for i in range(len(texts))
            ]
            ids = [f"{org_id}-{i}" for i in range(len(texts))]

            text_result = text_result + texts
            metadata_result = metadata_result + metadatas
            ids_result = ids_result + ids

    vectordb.add_texts(text_result, metadata_result, ids_result)
    vectordb.persist()
    print("VectorDB persisted successfully!")


org_roam_vectordb()
