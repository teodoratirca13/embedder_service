import time
from typing import Optional

import httpx

from fastapi_app.utils import get_settings, setup_logger

logger = setup_logger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 10.0


class SpringBootCallbackClient:
    """
    Client HTTP care notifica Spring Boot despre statusul final al job-ului
    de procesare a imaginilor (PATCH .../image-status).

    Presupunere de payload (necofirmata inca printr-un contract oficial —
    vezi Embedder_Async_Images_Contract_SpringBoot.md, sectiunea 10 din plan):

        {
            "document_id": 123,
            "status": "INDEXED" | "FAILED_IMAGES",
            "images_indexed": 5,
            "images_failed": 1
        }

    Esecul notificarii NU trebuie sa opreasca sau sa faca sa crape job-ul de
    fundal — se incearca de MAX_RETRIES ori, cu pauza intre incercari, si
    daca tot esueaza se logheaza eroarea si se intoarce False. Apelantul
    (image_pipeline.py) decide ce face mai departe (in principiu, nimic —
    documentul ramane INDEXED_TEXT_ONLY, profesorul poate reindexa manual).
    """

    def __init__(self, callback_url: str, username: str, password: str):
        if not callback_url:
            error_msg = "SPRING_BOOT_CALLBACK_URL nu este configurat — nu pot initializa SpringBootCallbackClient"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.callback_url = callback_url  #destinatia catre care se trimit datele
        self.auth = (username, password)
        self.client = httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) #instrumentrul care face trimiterea

        logger.info(
            "SpringBoot callback client initialized",
            extra={"extra_data": {"callback_url": callback_url}}
        )

    def notify_image_status(
            self,
            document_id: int,
            status: str,
            images_indexed: int,
            images_failed: int
    ) -> bool:
        """
        Trimite PATCH catre Spring Boot cu statusul final al job-ului de imagini.

        Args:
            document_id: id-ul documentului procesat
            status: "INDEXED" (toate imaginile ok, sau nu au existat imagini)
                    sau "FAILED_IMAGES" (Gemini complet indisponibil / toate
                    imaginile au esuat)
            images_indexed: cate imagini au fost captionate + upsertate cu succes
            images_failed: cate imagini au esuat (Gemini indisponibil, corupte etc.)

        Returns:
            True daca notificarea a reusit, False daca au esuat toate incercarile.
            Nu arunca exceptii.
        """
        payload = {
            "document_id": document_id,
            "status": status,
            "images_indexed": images_indexed,
            "images_failed": images_failed
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.patch(
                    self.callback_url,
                    json=payload,
                    auth=self.auth
                )
                response.raise_for_status()

                logger.info(
                    "SpringBoot notified successfully",
                    extra={"extra_data": {"document_id": document_id, "status": status, "attempt": attempt}}
                )
                return True

            except Exception as e:
                logger.warning(
                    "SpringBoot callback attempt failed",
                    extra={"extra_data": {
                        "document_id": document_id,
                        "attempt": attempt,
                        "max_retries": MAX_RETRIES,
                        "error": str(e)
                    }}
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)

        logger.error(
            "SpringBoot callback failed after all retries — document stays INDEXED_TEXT_ONLY",
            extra={"extra_data": {"document_id": document_id, "status": status}}
        )
        return False


_callback_client: Optional[SpringBootCallbackClient] = None


def get_springboot_callback_client() -> SpringBootCallbackClient:
    """
    Get or create the SpringBoot callback client singleton.

    Fail-fast doar la utilizare: daca SPRING_BOOT_CALLBACK_URL lipseste,
    eroarea apare aici (la prima folosire, in job-ul de fundal), nu la
    pornirea serviciului.
    """
    global _callback_client
    if _callback_client is None:
        settings = get_settings()
        _callback_client = SpringBootCallbackClient(
            callback_url=settings.spring_boot_callback_url,
            username=settings.spring_boot_callback_username,
            password=settings.spring_boot_callback_password
        )
    return _callback_client