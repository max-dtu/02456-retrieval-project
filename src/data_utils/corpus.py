import os
import shutil
import json
from datasets import Dataset, DatasetDict, load_from_disk
from src.data_utils.download_retrieval_datasets import get_fiqa_dataset, get_squad_dataset

class Corpus:
    def __init__(self, dataset_name: str, base_dir: str):
        self.dataset_name = dataset_name
        if not os.path.exists(f'{base_dir}/{self.dataset_name}'):
            raw_data_dir = f'{base_dir}/raw'
            os.makedirs(raw_data_dir, exist_ok=True)
            # load and process data
            self.ds = self.__load_data(base_dir=raw_data_dir)
            self.__rename_columns()
            self.relevant_ids = self.__assign_ids()
            os.makedirs(f'{base_dir}/{self.dataset_name}', exist_ok=True)
            DatasetDict(self.ds).save_to_disk(f'{base_dir}/{self.dataset_name}')
            with open(f'{base_dir}/{self.dataset_name}/relevant_ids.json', 'w') as f:
                json.dump(self.relevant_ids, f)
            
            print(f'Saved {self.dataset_name} dataset to path {base_dir}/{self.dataset_name}.')
            shutil.rmtree(f'{base_dir}/raw')
        else:
            self.ds = load_from_disk(f'{base_dir}/{self.dataset_name}')
            with open(f'{base_dir}/{self.dataset_name}/relevant_ids.json', 'r') as f:
                self.relevant_ids = json.load(f)


    @property
    def train(self):
        return self.ds.get('train', None)
    
    @property
    def val(self):
        return self.ds.get('validation', None)

    @property
    def test(self):
        return self.ds.get('test', None)

    @property
    def all_documents(self):
        splits = [self.train, self.val]
        if self.test is not None:
            splits.append(self.test)

        all_text = []
        all_ids = []
        seen_ids = set()

        for split in splits:
            for text_list, id_list in zip(split['text'], split['document_id']):
                for doc_text, doc_id in zip(text_list, id_list):
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_text.append([doc_text])
                        all_ids.append([doc_id])

        return Dataset.from_dict({'text': all_text,
                                  'document_id': all_ids})


    def __load_data(self, base_dir) -> dict[Dataset]:
        assert self.dataset_name in ['squad', 'fiqa'], f'Dataset {self.dataset_name} not supported. Options are ["squad", "fiqa"]'
        
        if self.dataset_name == 'fiqa':
            ds = get_fiqa_dataset(base_dir)

        elif self.dataset_name == 'squad':
            ds = get_squad_dataset(base_dir)

        return {key: ds[key] for key in ds.keys()}

    def __rename_columns(self) -> dict[Dataset]:
        column_mapping = {'fiqa':  'ground_truths',
                          'squad': 'context'
                         }
        
        for key in self.ds:
            self.ds[key] = self.ds[key].rename_column(column_mapping[self.dataset_name], 'text')
            if self.dataset_name == 'squad':
                self.ds[key] = self.ds[key].map(lambda row: {'text': [row['text']]})
        return self.ds

    def __assign_ids(self):
        q_id_counter = 0
        doc_id_counter = 0
        doc_text_to_id = {}
        relevant_document_ids = {}

        for key in self.ds:
            query_ids = []
            document_ids = []
            texts = self.ds[key]['text']
            for documents in texts:
                q_id = f'q{q_id_counter}'
                query_ids.append(q_id)
                relevant_ids = []
                for doc in documents:
                    if doc not in doc_text_to_id:
                        doc_id = f'doc{doc_id_counter}'
                        doc_text_to_id[doc] = doc_id
                        doc_id_counter += 1
                    else:
                        doc_id = doc_text_to_id[doc]
                    relevant_ids.append(doc_id)
                
                document_ids.append(relevant_ids)
                relevant_document_ids[q_id] = list(dict.fromkeys(relevant_ids))
                q_id_counter += 1
                
            self.ds[key] = self.ds[key].add_column('query_id', query_ids)
            self.ds[key] = self.ds[key].add_column('document_id', document_ids)

        return relevant_document_ids
            