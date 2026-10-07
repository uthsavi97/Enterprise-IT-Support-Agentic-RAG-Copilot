import time

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


settings = get_settings()


_embeddings = None
_vectorstore = None


# Free local embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MiniLM embedding dimension
EMBEDDING_DIMENSION = 384


def get_embedding_dimension() -> int:
    return EMBEDDING_DIMENSION


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    return _embeddings


def ensure_index():

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(
        api_key=settings.pinecone_api_key
    )

    # Get existing indexes
    names = [
        x["name"]
        for x in pc.list_indexes()
    ]

    # Check if our index already exists
    if settings.pinecone_index_name in names:

        index_info = pc.describe_index(
            settings.pinecone_index_name
        )

        current_dimension = getattr(
            index_info,
            "dimension",
            None
        )

        if current_dimension is None and isinstance(
            index_info,
            dict
        ):
            current_dimension = index_info.get(
                "dimension"
            )

        # Recreate index if dimension is wrong
        if (
            current_dimension is not None
            and current_dimension != desired_dimension
        ):

            print(
                f"Dimension mismatch: "
                f"Pinecone={current_dimension}, "
                f"Embedding={desired_dimension}"
            )

            print("Deleting old Pinecone index...")

            pc.delete_index(
                name=settings.pinecone_index_name
            )

            while (
                settings.pinecone_index_name
                in [
                    x["name"]
                    for x in pc.list_indexes()
                ]
            ):
                time.sleep(1)

    # Refresh index list
    names = [
        x["name"]
        for x in pc.list_indexes()
    ]

    # Create index if it doesn't exist
    if settings.pinecone_index_name not in names:

        print("Creating Pinecone index...")

        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

        # Wait until Pinecone index is ready
        while not pc.describe_index(
            settings.pinecone_index_name
        ).status["ready"]:
            time.sleep(1)

        print("Pinecone index is ready.")

    return pc.Index(
        settings.pinecone_index_name
    )


def get_vectorstore():

    global _vectorstore

    if _vectorstore is None:

        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


def get_retriever():

    return get_vectorstore().as_retriever(
        search_kwargs={
            "k": settings.top_k
        }
    )


def add_documents(chunks):

    store = get_vectorstore()

    return store.add_documents(chunks)