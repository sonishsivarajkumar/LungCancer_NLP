# data_split.py

import os
import random
import xml.etree.ElementTree as ET

def get_xml_files_with_annotations(base_folders_and_subfolders):
    xml_files = []
    for base_folder, folders in base_folders_and_subfolders:
        for folder_name in folders:
            folder_path = os.path.join(base_folder, folder_name)
            for xml_file in os.listdir(folder_path):
                if xml_file.endswith('.xml'):
                    xml_path = os.path.join(folder_path, xml_file)
                    # Check if the XML file has annotations
                    with open(xml_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    try:
                        tree = ET.ElementTree(ET.fromstring(content))
                        root = tree.getroot()
                        tags_element = root.find('TAGS')
                        if tags_element is not None and len(tags_element) > 0:
                            # The file has annotations
                            xml_files.append(xml_path)
                    except ET.ParseError as e:
                        print(f"Error parsing {xml_file}: {e}")
    return xml_files

def main():
    # Base folders and their respective subfolders for both annotators
    base_folder_subhash = r"C:\Users\sivarajkumars2\OneDrive - UPMC\Documents\Lung_cancer_project\Annotation_shared\LC_Annotations\Annotated Files\Round_4\Subhash"
    folders_subhash = [
        "Completed Partial Response",
        "Completed Progressive Disease",
        "Completed RECIST complete response",
        "Completed Stable Disease",
    ]
    
    base_folder_david = r"C:\Users\sivarajkumars2\OneDrive - UPMC\Documents\Lung_cancer_project\Annotation_shared\LC_Annotations\Annotated Files\Round_4\David_annotated_files_round_4\David's Done"
    folders_david = [
        "David Complete Response- Done",
        "David Partial Response- Done",
        "David progressive disease- Done",
        "David stable disease- done",
    ]
    
    base_folders_and_subfolders = [
        (base_folder_subhash, folders_subhash),
        (base_folder_david, folders_david),
    ]
    
    # Get all XML files with annotations
    xml_files = get_xml_files_with_annotations(base_folders_and_subfolders)
    print(f"Total XML files with annotations: {len(xml_files)}")
    
    # Perform a 70-30 split
    random.seed(42)
    random.shuffle(xml_files)
    split_index = int(len(xml_files) * 0.7)
    train_files = xml_files[:split_index]
    test_files = xml_files[split_index:]
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    
    # Save the lists to files
    os.makedirs('data', exist_ok=True)
    with open('data/train_files.txt', 'w') as f:
        for file in train_files:
            f.write(file + '\n')
    with open('data/test_files.txt', 'w') as f:
        for file in test_files:
            f.write(file + '\n')

if __name__ == "__main__":
    main()
