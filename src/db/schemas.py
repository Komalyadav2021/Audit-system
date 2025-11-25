# src/db/schemas.py
"""
Schemas (informational) for MongoDB collections:

documents:
{
  _id, filename, text, chunks: [{chunk_id, text, start, end}],
  uploaded_at, metadata: {source, pages, filetype}, version
}

labeled_documents:
{
  _id, document_id (ObjectId), labels: [ {line_id, date, description, debit, credit, balance, category} ],
  generated_at, labeler_version
}

agents_logs:
{ agent, request_id, input, output, confidence, timestamp }

qapairs:
{ question, answer, source_doc_id, created_at }

fine_tunes:
{ base_model, peft_config, dataset_path, metrics: {loss_history,..}, output_dir, created_at }
"""
