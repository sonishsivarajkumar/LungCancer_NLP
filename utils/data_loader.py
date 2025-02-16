# utils/data_loader.py


import os
import pandas as pd
import xml.etree.ElementTree as ET

def load_data_from_files(xml_files):
    data = []
    for xml_path in xml_files:
        # Extract file name for reference
        xml_file = os.path.basename(xml_path)
        # Parse XML and extract entities and annotations
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Use XML parser to extract data
        try:
            tree = ET.ElementTree(ET.fromstring(content))
            root = tree.getroot()
            # Extract text
            text_element = root.find('TEXT')
            if text_element is not None and text_element.text is not None:
                text = text_element.text
            else:
                continue
            # Extract tags
            tags_element = root.find('TAGS')
            if tags_element is not None:
                for child in tags_element:
                    # Process each entity
                    entity_type = child.tag
                    entity_id = child.attrib.get('id', '')
                    spans = child.attrib.get('spans', '')
                    text_span = child.attrib.get('text', '')
                    attributes = child.attrib
                    # Process spans
                    for span in spans.split(';'):
                        if '~' in span:
                            start, end = map(int, span.split('~'))
                            data.append({
                                'file_name': xml_file,
                                'text': text,
                                'entity_text': text_span,
                                'start': start,
                                'end': end,
                                'entity_type': entity_type,
                                'attributes': attributes,
                            })
        except ET.ParseError as e:
            print(f"Error parsing {xml_file}: {e}")
    df = pd.DataFrame(data)
    return df


# utils/data_loader.py
# utils/data_loader.py
# utils/data_loader.py
# utils/data_loader.py
# utils/data_loader.py

import os
import pandas as pd
import xml.etree.ElementTree as ET
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from itertools import combinations

def load_data_from_files_with_relations(xml_files):
    """
    Load data from XML files and extract relations based on Treatment_Response tags.

    Parameters:
        xml_files (list): List of XML file paths.

    Returns:
        pd.DataFrame: DataFrame containing extracted relations.
    """
    data = []
    for xml_path in xml_files:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        try:
            tree = ET.ElementTree(ET.fromstring(content))
            root = tree.getroot()
            print(f"Processing file: {xml_path}")
            print(f"Root tag: {root.tag}")
            print("Child tags under root:")
            for child in root:
                print(f"  {child.tag}")

            # Extract text from 'TEXT' element
            text_element = root.find('TEXT')
            if text_element is None or text_element.text is None:
                print(f"No 'TEXT' element found or empty in {xml_path}")
                continue
            text = text_element.text

            # Extract annotations from 'TAGS' element
            tags_element = root.find('TAGS')
            if tags_element is None:
                print(f"No 'TAGS' element found in {xml_path}")
                continue

            # Extract entities
            entities = {}
            for tag in tags_element.findall('*'):
                tag_name = tag.tag  # The tag name is the entity type
                # Skip relation tags
                if tag_name == 'Treatment_Response':
                    continue
                ann_id = tag.attrib.get('id', None)
                if not ann_id:
                    print(f"Entity without 'id' in {xml_path}, skipping.")
                    continue
                entity_type = tag_name  # Use the tag name as entity_type
                spans = tag.attrib.get('spans', '')
                text_span = tag.attrib.get('text', '')
                if not spans:
                    print(f"No 'spans' attribute for entity {ann_id} in {xml_path}, skipping.")
                    continue
                # Handle multiple spans
                span_list = spans.strip().split(';')
                for span_str in span_list:
                    offset_parts = span_str.strip().split('~')
                    if len(offset_parts) != 2:
                        print(f"Invalid span format '{span_str}' for entity {ann_id} in {xml_path}, skipping.")
                        continue
                    try:
                        start = int(offset_parts[0])
                        end = int(offset_parts[1])
                        entity_text = text[start:end]
                    except ValueError:
                        print(f"Non-integer span values '{span_str}' for entity {ann_id} in {xml_path}, skipping.")
                        continue

                    entities[ann_id] = {
                        'id': ann_id,
                        'entity_text': entity_text.strip(),
                        'entity_type': entity_type,
                        'start': start,
                        'end': end,
                        'file_name': os.path.basename(xml_path),
                        'text': text
                    }

            print(f"Number of entities extracted: {len(entities)}")
            if entities:
                entity_types = set([ent['entity_type'] for ent in entities.values()])
                print(f"Entity types in {os.path.basename(xml_path)}: {entity_types}")
            else:
                print(f"No entities extracted from {xml_path}.")

            # Extract relations
            relations_data = []

            # Collect positive relations
            positive_pairs = set()
            for rel in tags_element.findall('Treatment_Response'):
                rel_id = rel.attrib.get('id', 'Unknown_Relation_ID')
                arg0_id = rel.attrib.get('arg0ID')
                arg1_id = rel.attrib.get('arg1ID')
                rel_type = rel.tag  # e.g., 'Treatment_Response'

                if not arg0_id or not arg1_id:
                    print(f"Relation {rel_id} missing 'arg0ID' or 'arg1ID' in {xml_path}, skipping.")
                    continue

                ent0 = entities.get(arg0_id)
                ent1 = entities.get(arg1_id)

                if not ent0 or not ent1:
                    print(f"Relation {rel_id} references unknown entities '{arg0_id}' or '{arg1_id}' in {xml_path}, skipping.")
                    continue

                # Extract context
                if ent0['end'] <= ent1['start']:
                    context = text[ent0['end']:ent1['start']].strip()
                else:
                    context = text[ent1['end']:ent0['start']].strip()

                relations_data.append({
                    'file_name': os.path.basename(xml_path),
                    'ent1_text': ent0['entity_text'],
                    'ent1_type': ent0['entity_type'],
                    'ent2_text': ent1['entity_text'],
                    'ent2_type': ent1['entity_type'],
                    'context': context,
                    'relation': rel_type  # e.g., 'Treatment_Response'
                })

                # Add to positive pairs
                positive_pairs.add((arg0_id, arg1_id))

            # Generate negative examples
            entity_ids = list(entities.keys())
            all_pairs = set(combinations(entity_ids, 2))
            negative_pairs = all_pairs - positive_pairs

            for arg0_id, arg1_id in negative_pairs:
                ent0 = entities[arg0_id]
                ent1 = entities[arg1_id]
                # Extract context
                if ent0['end'] <= ent1['start']:
                    context = text[ent0['end']:ent1['start']].strip()
                else:
                    context = text[ent1['end']:ent0['start']].strip()
                relations_data.append({
                    'file_name': os.path.basename(xml_path),
                    'ent1_text': ent0['entity_text'],
                    'ent1_type': ent0['entity_type'],
                    'ent2_text': ent1['entity_text'],
                    'ent2_type': ent1['entity_type'],
                    'context': context,
                    'relation': 'No_Relation'
                })

            print(f"Number of relations inferred: {len(relations_data)}")

            data.extend(relations_data)
        except ET.ParseError as e:
            print(f"Error parsing {xml_path}: {e}")
            continue
    print(f"Total number of relations: {len(data)}")
    if data:
        df = pd.DataFrame(data)
        print(f"Columns in df: {df.columns.tolist()}")
    else:
        df = pd.DataFrame()
        print("No relations extracted. DataFrame is empty.")
    return df
