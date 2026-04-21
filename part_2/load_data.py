import os
import random
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast

PAD_IDX = 0

AIRLINE_CODE_TO_NAME = {
    'AA': ['american airlines', 'american'],
    'AC': ['air canada'],
    'CO': ['continental'],
    'DL': ['delta'],
    'LH': ['lufthansa', 'lufthansa airlines'],
    'ML': ['midway airlines', 'midway'],
    'NW': ['northwest', 'northwest airlines'],
    'TW': ['twa'],
    'UA': ['united', 'united airlines'],
    'US': ['us air', 'usair'],
    'WN': ['southwest airlines', 'southwest'],
    'YX': ['midwest express', 'midwest express airlines'],
}

class T5Dataset(Dataset):
    def __init__(
        self,
        data_folder,
        split,
        augment=False,
        augment_ratio=0.0,
        augment_seed=42,
        augment_num_augments=1,
    ):
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data_folder = data_folder
        self.split = split
        self.augment = augment
        self.augment_ratio = augment_ratio
        self.augment_seed = augment_seed
        self.augment_num_augments = augment_num_augments
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        with open(os.path.join(data_folder, f"{split}.nl"), 'r') as f:
            nls = [line.strip() for line in f.readlines()]
        
        if split != 'test':
            with open(os.path.join(data_folder, f"{split}.sql"), 'r') as f:
                sqls = [line.strip() for line in f.readlines()]
            paired = list(zip(nls, sqls))
            if split == 'train' and self.augment and self.augment_ratio > 0:
                rng = random.Random(self.augment_seed)
                aug_nls, aug_sqls = self.augment_entity_swap(
                    nls,
                    sqls,
                    rng,
                    num_augments=self.augment_num_augments,
                    apply_prob=self.augment_ratio,
                )
                paired = list(zip(aug_nls, aug_sqls))
                print(f"[DataAug] train examples: original={len(nls)}, augmented={len(aug_nls) - len(nls)}, total={len(paired)}")
            return paired
        else:
            return nls

    def augment_entity_swap(self, nl_lines, sql_lines, rng, num_augments=1, apply_prob=1.0):
        all_cities = sorted(
            set(
                city
                for sql in sql_lines
                for city in re.findall(r"city_name\s*=\s*'([^']+)'", sql)
            )
        )
        all_airline_codes = sorted(
            set(
                code
                for sql in sql_lines
                for code in re.findall(r"airline_code\s*=\s*'([^']+)'", sql)
                if code in AIRLINE_CODE_TO_NAME
            )
        )

        aug_nl, aug_sql = list(nl_lines), list(sql_lines)

        for nl, sql in zip(nl_lines, sql_lines):
            if rng.random() > apply_prob:
                continue

            sql_cities = list(dict.fromkeys(re.findall(r"city_name\s*=\s*'([^']+)'", sql)))
            sql_airlines = list(
                dict.fromkeys(
                    code
                    for code in re.findall(r"airline_code\s*=\s*'([^']+)'", sql)
                    if code in AIRLINE_CODE_TO_NAME
                )
            )

            has_cities = bool(sql_cities)
            has_airlines = bool(sql_airlines)
            if not has_cities and not has_airlines:
                continue

            for _ in range(num_augments):
                new_sql = sql
                new_nl = nl

                if has_cities:
                    available_cities = [c for c in all_cities if c not in sql_cities]
                    if len(available_cities) < len(sql_cities):
                        continue
                    city_map = dict(zip(sql_cities, rng.sample(available_cities, len(sql_cities))))
                    for old_city, new_city in city_map.items():
                        new_sql = new_sql.replace(f"'{old_city}'", f"'{new_city}'")
                        new_nl = re.sub(
                            re.escape(old_city),
                            new_city.lower(),
                            new_nl,
                            flags=re.IGNORECASE,
                        )

                if has_airlines:
                    available_airlines = [c for c in all_airline_codes if c not in sql_airlines]
                    if len(available_airlines) < len(sql_airlines):
                        continue
                    airline_map = dict(zip(sql_airlines, rng.sample(available_airlines, len(sql_airlines))))
                    for old_code, new_code in airline_map.items():
                        new_sql = re.sub(
                            rf"airline_code\s*=\s*'{re.escape(old_code)}'",
                            f"airline_code = '{new_code}'",
                            new_sql,
                        )
                        old_names = sorted(AIRLINE_CODE_TO_NAME[old_code], key=len, reverse=True)
                        new_name = AIRLINE_CODE_TO_NAME[new_code][0]
                        replaced = False
                        for old_name in old_names:
                            if re.search(re.escape(old_name), new_nl, flags=re.IGNORECASE):
                                new_nl = re.sub(
                                    re.escape(old_name),
                                    new_name,
                                    new_nl,
                                    flags=re.IGNORECASE,
                                    count=1,
                                )
                                replaced = True
                                break
                        if not replaced:
                            new_nl = re.sub(
                                rf"\b{re.escape(old_code)}\b",
                                new_code.lower(),
                                new_nl,
                                flags=re.IGNORECASE,
                            )

                if new_nl != nl or new_sql != sql:
                    aug_nl.append(new_nl)
                    aug_sql.append(new_sql)

        return aug_nl, aug_sql

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.split != 'test':
            nl, sql = self.data[idx]
            nl_enc = self.tokenizer(nl, return_tensors='pt', truncation=True, max_length=512)
            sql_enc = self.tokenizer(sql, return_tensors='pt', truncation=True, max_length=512)
            # Add BOS token to decoder input (using pad_token_id or extra_id)
            decoder_input = torch.cat([torch.tensor([[self.tokenizer.pad_token_id]]), sql_enc.input_ids[:, :-1]], dim=-1)
            return nl_enc.input_ids[0], nl_enc.attention_mask[0], decoder_input[0], sql_enc.input_ids[0], decoder_input[0][0]
        else:
            nl = self.data[idx]
            nl_enc = self.tokenizer(nl, return_tensors='pt', truncation=True, max_length=512)
            initial_decoder_input = torch.tensor([self.tokenizer.pad_token_id])
            return nl_enc.input_ids[0], nl_enc.attention_mask[0], initial_decoder_input

def normal_collate_fn(batch):
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([item[3] for item in batch], batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack([item[4] for item in batch])
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
    initial_decoder_inputs = torch.stack([item[2] for item in batch])
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, augment_train=False, augment_ratio=0.0, augment_seed=42, augment_num_augments=1):
    data_folder = 'data'
    use_augmentation = split == "train" and augment_train and augment_ratio > 0
    dset = T5Dataset(
        data_folder,
        split,
        augment=use_augmentation,
        augment_ratio=augment_ratio,
        augment_seed=augment_seed,
        augment_num_augments=augment_num_augments,
    )
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    # 增加num_workers
    num_workers = 8 if split == "train" else 2
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                          num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    return dataloader

def load_t5_data(
    batch_size,
    test_batch_size,
    augment_train=False,
    augment_ratio=0.0,
    augment_seed=42,
    augment_num_augments=1,
):
    return (
        get_dataloader(
            batch_size,
            "train",
            augment_train=augment_train,
            augment_ratio=augment_ratio,
            augment_seed=augment_seed,
            augment_num_augments=augment_num_augments,
        ),
        get_dataloader(test_batch_size, "dev"),
        get_dataloader(test_batch_size, "test"),
    )
