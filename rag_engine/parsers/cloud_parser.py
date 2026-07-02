# ==========================================
# File: rag_engine/parsers/cloud_parser.py
# ==========================================

import asyncio
import os

from llama_cloud import LlamaCloud

from ..core.logger import get_logger


class UniversalParser:
    def __init__(self, config):
        self.log = get_logger("UniversalParser")

        self.client = LlamaCloud(
            api_key=config.get("LLAMA_CLOUD_API_KEY")
        )

        self.tier = config.get("PARSER_TIER", "agentic")
        self.version = config.get("PARSER_VERSION", "latest")

    async def _parse_single_file(self, path: str):
        self.log.info(f"Parsing {path}")

        result = await asyncio.to_thread(
            self.client.parsing.parse,
            upload_file=path,
            tier=self.tier,
            version=self.version,
            expand=["items"],
            verbose=True,
        )

        data = result.items.model_dump()

        final_pages = []

        for page in data["pages"]:

            page_content = []

            for item in page["items"]:

                item_type = item.get("type", "")

                # -----------------------------
                # Skip useless repeated content
                # -----------------------------
                if item_type == "footer":
                    continue

                value = item.get("value", "")

                if (
                    item_type == "text"
                    and isinstance(value, str)
                    and "logo" in value.lower()
                ):
                    continue

                # -----------------------------
                # Headings
                # -----------------------------
                if item_type == "heading":

                    page_content.append(item.get("md", value))

                # -----------------------------
                # Tables
                # -----------------------------
                elif item_type == "table":

                    page_content.append(item.get("md", ""))

                # -----------------------------
                # Normal Text
                # -----------------------------
                elif item_type == "text":

                    page_content.append(value)

                # -----------------------------
                # Diagrams (Future Compatible)
                # -----------------------------
                elif item_type == "diagram":

                    if item.get("mermaid"):

                        page_content.append(
                            f"```mermaid\n{item['mermaid']}\n```"
                        )

                    elif value:

                        page_content.append(value)

                # -----------------------------
                # Images
                # -----------------------------
                elif item_type == "image":

                    if value:

                        page_content.append(
                            f"Image Description:\n{value}"
                        )

                # -----------------------------
                # Unknown future types
                # -----------------------------
                else:

                    if item.get("md"):

                        page_content.append(item["md"])

                    elif value:

                        page_content.append(value)

            content = "\n\n".join(page_content).strip()

            if not content:
                continue

            metadata = {
                "source_path": path,
                "file_name": os.path.basename(path),
                "page_number": page["page_number"],
                "file_type": os.path.splitext(path)[1].replace(".", "").lower(),
            }

            final_pages.append(
                {
                    "content": content,
                    "metadata": metadata,
                }
            )

        return final_pages

    async def parse_files(self, file_paths):
        self.log.info(
            f"Starting Visual Parsing for {len(file_paths)} files..."
        )

        try:

            final_data = []

            for path in file_paths:

                pages = await self._parse_single_file(path)

                final_data.extend(pages)

            self.log.info(
                f"Successfully processed {len(final_data)} pages."
            )

            return final_data

        except Exception as e:

            self.log.exception(
                f"Visual Parsing failed: {e}"
            )

            return None