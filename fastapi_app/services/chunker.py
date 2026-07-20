from langchain_text_splitters import RecursiveCharacterTextSplitter

from fastapi_app.utils import setup_logger

logger = setup_logger(__name__)


class TextChunker:

    def __init__(
            self,
            chunk_size: int = 800,
            chunk_overlap: int = 150,
            min_chunk_length: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length


        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )

        logger.info(
            "TextChunker initialized",
            extra={"extra_data": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "min_chunk_length": min_chunk_length
            }}
        )

    def chunk_text(self, text: str) -> list[str]: #return list of chunks
        logger.info(
            "Starting text chunking",
            extra={"extra_data": {"text_length": len(text)}}
        )

        # Split using LangChain
        chunks = self.splitter.split_text(text)
        logger.info(
            "Initial split complete",
            extra={"extra_data": {"initial_chunk_count": len(chunks)}}
        )

        # scot chunk urile prea mici
        filtered_chunks = [
            chunk for chunk in chunks
            if len(chunk.strip()) >= self.min_chunk_length
        ]

        filtered_count = len(chunks) - len(filtered_chunks)
        if filtered_count > 0:
            logger.info(
                "Filtered out short chunks",
                extra={"extra_data": {"filtered_count": filtered_count}}
            )

        logger.info(
            "Chunking complete",
            extra={"extra_data": {
                "final_chunk_count": len(filtered_chunks),
                "avg_chunk_length": sum(len(c) for c in filtered_chunks) // len(
                    filtered_chunks) if filtered_chunks else 0
            }}
        )

        return filtered_chunks


# Create a singleton instance
_chunker = None


def get_text_chunker() -> TextChunker:
    global _chunker
    if _chunker is None:
        _chunker = TextChunker(
            chunk_size=800,
            chunk_overlap=150,
            min_chunk_length=50
        )
    return _chunker