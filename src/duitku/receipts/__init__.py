"""Receipt photos: file storage, vision extraction, validation.

Images live on disk only. Their bytes (and base64) exist transiently inside the outgoing vision
request; they never reach graph state, checkpoints, tool results, logs or spans.
"""
