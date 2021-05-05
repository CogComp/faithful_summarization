from stanza.models.common.doc import Document, Span
from quantity_util import quantity
import json
import random
from typing import List


def _get_non_lower_case_tokens(surface: str) -> List[str]:
    toks = surface.split()
    valid_toks = [tok for tok in toks if not tok.islower()]
    return valid_toks


NORM_NUM_TYPE = {"DATE", "TIME", "PERCENT", "QUANTITY", "CARDINAL"} # We dont normalize MONEY and ORDINAL


def check_hallucination(source_ents: List[Span], target_ents: List[Span]) -> List[int]:
    """
    Check if any of the entites in the summary have never appeared in the original text.
    :return: List of indices of hallucinated entities in target_ents, empty list if no hallucination is found
    """
    s_ent_toks_by_type = {}

    hallucinated_ents = []

    if len(target_ents) == 0:
        return hallucinated_ents

    for s_ent in source_ents:
        non_lower_toks = _get_non_lower_case_tokens(s_ent.text)
        if len(non_lower_toks) > 0:
            if s_ent.type not in s_ent_toks_by_type:
                s_ent_toks_by_type[s_ent.type] = set(non_lower_toks)
            else:
                s_ent_toks_by_type[s_ent.type].update(non_lower_toks)

    for idx, t_ent in enumerate(target_ents):
        non_lower_toks = _get_non_lower_case_tokens(t_ent.text)
        if t_ent.type in s_ent_toks_by_type:
            source_non_lower_toks = s_ent_toks_by_type[t_ent.type]
            for _tok in non_lower_toks:
                # If the target entity contains non-lower case token that never appeared in source text
                if _tok not in source_non_lower_toks:
                    hallucinated_ents.append(idx)
                    break
        else:
            hallucinated_ents.append(idx)
    return hallucinated_ents


TIME_TYPE = {"TIME", "DATE"}


def is_cardinal_different(entity1_toks: List[str],
                          entity2_toks: List[str]):
    try:
        norm1 = quantity(entity1_toks)
        norm2 = quantity(entity2_toks)
        if (norm1 is not None) and (norm2 is not None) and norm1 == norm2:
            return False
        else:
            return True

    except Exception as e:
        return True


def is_entity_different(entity1_surface: str,
                        entity2_surface: str,
                        entity_type: str):
    ent1_toks = entity1_surface.split(" ")
    ent2_toks = entity2_surface.split(" ")

    if entity_type == "CARDINAL":
        return is_cardinal_different(ent1_toks, ent2_toks)
    else:
        _overlap = [tok for tok in ent1_toks if tok in ent2_toks]
        return len(_overlap) == 0


def make_perturbations(target_text: str,
                       target_ents: List[Span],
                       source_ents: List[Span],
                       is_training_mode: bool,
                       max_perturbation_per_example: int = 10):

    perturbations = []

    # Check for hallucinated entities in the target_text
    hallucinated_ent_idx = check_hallucination(source_ents=source_ents, target_ents=target_ents)

    # In training mode, we don't want to produce perturbations when there's hallucination present
    if is_training_mode and len(hallucinated_ent_idx) > 0:
        return perturbations, []

    # In test mode, we only want to produce perturbations on the hallucinated entities
    # We also want to ignore ORDINALs, as they are in most of the cases impossible to correct
    if not is_training_mode:
        target_ents = [ent for idx, ent in enumerate(target_ents) if idx in hallucinated_ent_idx]
        target_ents = [ent for ent in target_ents if ent.type != "ORDINAL"]

    change_list = []
    for tgt_ent in target_ents:
        _tgt_ent_surface = tgt_ent.text
        _type = tgt_ent.type
        _start_offset = tgt_ent.start_char
        _end_offset = tgt_ent.end_char

        compatible_src_ents = set([_ent for _ent in source_ents if _ent.type == _type])

        if is_training_mode:
            compatible_src_ents = [_ent for _ent in compatible_src_ents if is_entity_different(_tgt_ent_surface, _ent.text, _type)]

        seen_ents = set()

        for src_ent in compatible_src_ents:
            if src_ent.text not in seen_ents:
                _perturb_text = target_text[:_start_offset] + src_ent.text + target_text[_end_offset:]
                seen_ents.add(src_ent.text)
                change_list.append([_tgt_ent_surface, src_ent.text, _start_offset, _end_offset])
                perturbations.append(_perturb_text)

    if len(perturbations) > max_perturbation_per_example > 0:
        _z = list(zip(perturbations, change_list))
        _sample = random.choices(_z, k=max_perturbation_per_example)
        _uz = list(zip(*_sample))
        return list(_uz[0]), list(_uz[1])
    else:
        return perturbations, change_list


def make_data(source_nlp: str,
              source_file: str,
              target_nlp: str,
              target_file: str,
              output_path: str,
              is_training_mode: bool = True,
              max_perturbation_per_example: int = 10):

    print("Running in training mode (perturbations will be generated only on faithful summaries): {}".format(is_training_mode))

    count = 0
    with open(source_nlp) as source_nlp_file, \
         open(source_file) as source_file, \
         open(target_nlp) as target_nlp_file, \
         open(target_file) as target_file, \
         open(output_path, 'w') as out_file:
        for line in source_nlp_file:
            src_doc_dict = json.loads(line)
            if len(src_doc_dict) > 0:
                src_doc = Document(src_doc_dict)
            else:
                src_doc = None

            source_text = source_file.readline()

            # Read the corresponding target
            target_text = target_file.readline().strip()
            target_line = target_nlp_file.readline()
            tgt_doc_dict = json.loads(target_line)
            if len(tgt_doc_dict) > 0:
                tgt_doc = Document(tgt_doc_dict)
            else:
                tgt_doc = None

            if (src_doc is None) or (tgt_doc is None):
                example = {
                    "source_text": source_text,
                    "positive_examples": [target_text],
                    "negative_examples": []
                }

                out_file.write(json.dumps(example))
                out_file.write("\n")

                continue

            src_doc._text = source_text
            src_doc.build_ents()

            tgt_doc._text = target_text
            tgt_doc.build_ents()

            neg_examples, changed_list = make_perturbations(target_text=target_text,
                                                            target_ents=tgt_doc.ents,
                                                            source_ents=src_doc.ents,
                                                            is_training_mode=is_training_mode,
                                                            max_perturbation_per_example=max_perturbation_per_example)

            # De-deuplicate the ones that are the same as the original summary
            neg_examples = [neg for neg in neg_examples if neg != target_text]
            example = {
                "source_text": source_text,
                "positive_examples": [target_text],
                "negative_examples": neg_examples,
                "changed_list": changed_list
            }

            out_file.write(json.dumps(example))
            out_file.write("\n")
            count += 1

            if count % 500 == 0:
                print("Processed: {}".format(count))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare data for ')
    parser.add_argument('source_stanza_output', type=str, help="Path to stanza output for source text")
    parser.add_argument('source_file', type=str, help="Path to a line separated file for source text")
    parser.add_argument('target_stanza_output', type=str, help="Path to stanza output for target summaries")
    parser.add_argument('target_file', type=str, help="Path to a line separated file for target summaries")
    parser.add_argument('output_path', type=str, help="Path to output")
    parser.add_argument('--eval', action="store_true", help="Run the perturbations in evaluation mode")
    parser.add_argument('--limit', type=int, help="max number of perturbations generated per example", default=10)
    args = parser.parse_args()

    is_training_mode = True
    if args.eval:
        is_training_mode = not args.eval

    make_data(args.source_stanza_output,
              args.source_file,
              args.target_stanza_output,
              args.target_file,
              args.output_path,
              is_training_mode,
              args.limit)
