from typing import Dict, Type
import importlib
import pkgutil
import nltk
from pathlib import Path
from anonymizer.base import BaseFilter

def ensure_nltk_resources():
    """
    Download all required NLTK resources.
    """
    resources = [
        'punkt',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]

    # First, try direct downloads
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Failed to download NLTK resource {resource}: {e}")
            raise

    # Verify the tagger is available
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)

def load_filters() -> Dict[str, BaseFilter]:
    """
    Dynamically load all available filters from the filters directory.

    Returns:
        Dictionary mapping filter names to filter instances
    """
    filters = {}
    filters_dir = Path(__file__).parent / "filters"

    # Ensure NLTK resources are available
    ensure_nltk_resources()

    for module_info in pkgutil.iter_modules([str(filters_dir)]):
        if module_info.name.startswith('__'):
            continue

        module = importlib.import_module(f"anonymizer.filters.{module_info.name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                issubclass(attr, BaseFilter) and
                attr != BaseFilter):
                filter_name = module_info.name
                filters[filter_name] = attr()

    return filters