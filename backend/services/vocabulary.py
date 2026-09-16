"""
Vocabulary registry loader and lookup.
Single source of truth for what the system claims to support.
"""
import os
import json
from typing import Dict, Any, Optional, List

_registry: Dict[str, Any] = {}
_loaded = False


def load_registry(registry_path: str):
    """Load the vocabulary registry from disk."""
    global _registry, _loaded
    
    if not os.path.exists(registry_path):
        print(f"[WARN] Vocabulary registry not found at {registry_path}")
        _registry = _get_default_registry()
    else:
        with open(registry_path) as f:
            _registry = json.load(f)
    
    _loaded = True
    print(f"[OK] Vocabulary registry loaded")


def _get_default_registry() -> Dict[str, Any]:
    """Return a minimal default registry."""
    return {
        "schema_version": "1.0",
        "recognition": {
            "digits": {
                "model": "landmark_digits",
                "source_dataset": "ardamavi/Sign-Language-Digits-Dataset",
                "modality": "static",
                "classes": {str(i): {"label": str(i), "recognition_supported": True, "sources": ["image"]} for i in range(10)}
            },
            "words": {
                "model": "temporal_words",
                "source_dataset": "vidit031/isl-isolated-8words",
                "modality": "temporal",
                "classes": {}
            }
        },
        "production": {},
        "counter_phrases": [
            "Please wait for your turn.",
            "Please show your ID.",
            "Your token number is {N}.",
            "Go to counter {N}.",
            "Please sign here.",
            "Thank you. You may go."
        ]
    }


def get_registry() -> Dict[str, Any]:
    return _registry


def is_recognition_supported(token: str, mode: str = "static") -> bool:
    """Check if a token is supported for recognition."""
    if mode in ("letters", "landmark_letters"):
        section = "letters"
    elif mode in ("gesture_asl", "temporal_letters_phrases"):
        section = "gesture_asl"
    elif mode in ("temporal", "words", "temporal_words"):
        section = "words"
    else:
        section = "digits"
    classes = _registry.get("recognition", {}).get(section, {}).get("classes", {})
    entry = classes.get(token, {})
    return entry.get("recognition_supported", False)


def get_production_asset(token: str) -> Optional[Dict[str, Any]]:
    """Look up a token in the production (text-to-sign) section."""
    return _registry.get("production", {}).get(token.lower())


def get_counter_phrases() -> List[str]:
    return _registry.get("counter_phrases", [])


def text_to_sign_tokens(text: str) -> Dict[str, Any]:
    """
    Convert text to sign caption tokens.
    Normalizes input, expands numerals, looks up each token.
    Fingerspelling Fallback Tier:
      Any word not matching an existing whole-word sign decomposes into constituent letters
      if all letters exist as production assets (/signs/{letter}.png).
    """
    import re
    
    # Normalize
    text_lower = text.lower().strip()
    # Remove punctuation but keep spaces and numbers
    text_clean = re.sub(r'[^\w\s]', '', text_lower)
    
    # Split into tokens
    raw_tokens = text_clean.split()
    
    # Expand numbers into digit tokens
    expanded = []
    for token in raw_tokens:
        if token.isdigit():
            # Expand multi-digit numbers into individual digits
            for digit in token:
                expanded.append(digit)
        else:
            expanded.append(token)
    
    # Look up each token
    result_tokens = []
    matched = 0
    
    for token in expanded:
        prod = get_production_asset(token)
        
        if prod and prod.get("available", False):
            result_tokens.append({
                "token": token,
                "available": True,
                "asset": prod.get("caption_asset", ""),
                "kind": "digit" if token.isdigit() else "word"
            })
            matched += 1
        elif token.isalpha():
            # Fingerspelling fallback tier:
            # Check if all constituent letters exist in production assets
            letter_tokens = []
            all_letters_found = True
            for char in token:
                char_prod = get_production_asset(char)
                if char_prod and char_prod.get("available", False):
                    letter_tokens.append({
                        "token": char,
                        "available": True,
                        "asset": char_prod.get("caption_asset", f"/signs/{char}.png"),
                        "kind": "letter"
                    })
                else:
                    all_letters_found = False
                    break
            
            if all_letters_found and len(letter_tokens) > 0:
                result_tokens.extend(letter_tokens)
                matched += 1
            else:
                result_tokens.append({
                    "token": token,
                    "available": False,
                    "reason": "not_in_vocabulary"
                })
        else:
            result_tokens.append({
                "token": token,
                "available": False,
                "reason": "not_in_vocabulary"
            })
    
    total = len(expanded)
    coverage = matched / total if total > 0 else 0.0
    
    unmatched = total - matched
    if unmatched == 0:
        notice = "All words have sign captions."
    else:
        notice = f"{matched} of {total} words have sign captions. Unmatched words are shown as text only."
    
    return {
        "tokens": result_tokens,
        "coverage": round(coverage, 2),
        "notice": notice
    }


def validate_models_and_registry(model_labels: Dict[str, List[str]]) -> Dict[str, bool]:
    """
    Validate that loaded model labels match classes defined in registry.json.
    model_labels: {"digits": [...], "words": [...], "letters": [...], "gesture_asl": [...]}
    """
    results = {}
    recognition = _registry.get("recognition", {})
    
    for head_key, labels in model_labels.items():
        if not labels:
            results[head_key] = False
            continue
        head_classes = recognition.get(head_key, {}).get("classes", {})
        reg_classes = set(head_classes.keys())
        active_reg_classes = {k for k, v in head_classes.items() if v.get("recognition_supported", True)}
        model_classes = set(labels)
        # Check exact match against all classes or active recognition classes
        matches = (reg_classes == model_classes or 
                   active_reg_classes == model_classes or
                   reg_classes == {l.lower() for l in labels} or
                   active_reg_classes == {l.lower() for l in labels})
        results[head_key] = matches
        if matches:
            print(f"[OK] Startup Check: Head '{head_key}' matches registry ({len(labels)} classes)")
        else:
            print(f"[WARN] Startup Check: Head '{head_key}' mismatch between model ({len(labels)}) and registry ({len(reg_classes)})")
            
    return results
