# Investigative AI Core Modules

from .extract import (
    extract_from_document,
    load_extracted,
    save_extracted,
    merge_extraction,
    detect_conflicts,
    get_extraction_summary,
    remove_document_extraction
)

from .router import (
    classify_query,
    answer_exhaustive_query,
    should_use_extracted_data,
    get_query_category
)

from .coa import (
    decompose_query,
    should_expand_query,
    coa_report,
    coa_report_with_progress,
    stream_manager_response
)
