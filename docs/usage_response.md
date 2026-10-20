# Images response and usage

Scope: `/v1/images/generations` and `/v1/images/edits`.
The shared response builder matches SenseNova commit
`f3a2befe0902ce25d5784bdc540ff66ba70bcc3c`.

- Each image has `output_format`, determined from its actual encoded bytes.
- `usage.image_count` counts returned images.
- `usage.width` and `usage.height` describe the actual common output size.
- `usage.input_tokens_details` contains text and reference-image token counts.
- `input_tokens` and `total_tokens` equal text tokens only, excluding image tokens.
- Missing model token statistics remain unavailable; output pixels are not tokens.
- Repeated outputs for the same prompt do not multiply its input token count.

A result carries one image in `image_b64`, several in `images_b64`, and the
producer's statistics in `image_usage`; the client keeps only string images.

A UMM result reports the same statistics as one `usage` segment whose data has
`image_usage` and the producing `stage_name`. The final result fills
`image_usage` and `stage_name` from that segment. A streamed request received
the segment's data in the stream, so its final result leaves them unset.
Image segments carry `InlineImage` data, whose size and SHA-256 must match the
encoded bytes.

This change supplies the response builder and the usage carriers. Models
supply the producer and mount the Images API routes.

The previous Chat usage extensions, streaming usage options, and `/generate`
cached-token mapping are withdrawn. Existing base token usage is unchanged.
Metrics remain a separate branch and are outside this change.
