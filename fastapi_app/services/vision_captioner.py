from typing import Optional

from google import genai
from google.genai import types
from fastapi_app.utils import get_settings, setup_logger

logger = setup_logger(__name__)

CAPTION_PROMPT = (
    "Briefly describe, in English, what is shown in this image extracted "
    "from university course material (it could be a diagram, a chart, a "
    "schema, a photo, or a table). Focus on information useful to a student "
    "who would search for this content through a text question to a "
    "chatbot. Reply with only the description, without introductions like "
    "\"this image shows\"."
)


class VisionCaptioner:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            error_msg = "GEMINI_API_KEY nu este configurat — nu pot initializa VisionCaptioner"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.model = model

        try:
            self.client = genai.Client(api_key=api_key)  #clientul care stie sa cuminice cu API-ul GEMINI
            logger.info("Vision captioner initialized", extra={"extra_data": {"model": model}})
        except Exception as e:
            error_msg = f"Failed to initialize Gemini client: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def caption_image(self, image_bytes: bytes, mime_type: str = "image/png") -> Optional[str]:
        logger.debug(
            "Captioning image",
            extra={"extra_data": {"mime_type": mime_type, "size_bytes": len(image_bytes)}}
        )
        try:
            response = self.client.models.generate_content( #generate_content -> trimite catre gemini cererea
                model=self.model,
                contents=[             #imaginea care e deja in bytes     #tipul (png)
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    CAPTION_PROMPT,
                ],
            )
            caption = (response.text or "").strip()
            if not caption:
                logger.warning("Gemini returned empty caption — skipping image")
                return None
            logger.debug("Image captioned successfully", extra={"extra_data": {"caption_length": len(caption)}})
            return caption
        except Exception as e:
            logger.warning("Failed to caption image — skipping", extra={"extra_data": {"error": str(e), "mime_type": mime_type}})
            return None


_vision_captioner = None


def get_vision_captioner() -> VisionCaptioner:   #daca avem  il refoloseste, daca nu creaza instanta
    global _vision_captioner
    if _vision_captioner is None:
        settings = get_settings()
        _vision_captioner = VisionCaptioner(
            api_key=settings.gemini_api_key,
            model=settings.gemini_vision_model
        )
    return _vision_captioner