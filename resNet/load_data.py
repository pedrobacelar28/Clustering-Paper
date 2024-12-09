import h5py
import pandas as pd
import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import MinMaxScaler
import random


class MyDataloader:
    def __init__(
        self,
        h5_data,
        h5_ids,
        h5_indices,
        metadata,
        exam_id_col,
        output_cols,
        batch_size,
        data_dtype,
        output_dtype,
        world_size=1,
        rank=0,
        epoch_seed_mult=1,
        with_metadata=False,
    ):
        self.h5_data = h5_data
        self.h5_ids = h5_ids
        self.h5_indices = h5_indices

        self.metadata = metadata
        self.exam_id_col = exam_id_col
        self.output_cols = output_cols

        self.batch_size = batch_size

        self.data_dtype = data_dtype
        self.output_dtype = output_dtype

        self.distributed = world_size > 1
        self.world_size = world_size
        self.rank = rank
        self.epoch_seed_mult = epoch_seed_mult

        self.with_output = self.output_cols is not None
        self.with_metadata = with_metadata

        self.dataset_size = len(h5_indices)
        self.num_batches = int(
            np.ceil(self.dataset_size / (self.batch_size * world_size))
        )
        self.current_batch = 0
        self.current_epoch = 0

        if self.distributed:
            self.distributed_sampler = DistributedSampler(
                self, num_replicas=self.world_size, rank=self.rank
            )

    def __iter__(self):
        self.current_batch = 0

        if self.distributed:
            torch.manual_seed(self.current_epoch * self.epoch_seed_mult)
            self.distributed_sampler.set_epoch(self.current_epoch)
            self.indices = iter(self.distributed_sampler)

        else:
            np.random.seed(self.current_epoch * self.epoch_seed_mult)
            np.random.shuffle(self.h5_indices)
            self.indices = iter(range(len(self)))

        self.current_epoch += 1

        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        idx = next(self.indices)

        # Calculate start and end indices for the current batch
        start_idx = idx * self.batch_size
        end_idx = min(
            (start_idx + self.batch_size),
            self.dataset_size,
        )

        # Get the shuffled indices for the current batch
        batch_indices = self.h5_indices[start_idx:end_idx]

        # Sort the indices for h5py compatibility
        sorted_indices = np.sort(batch_indices)

        # Fetch the data using sorted indices
        sorted_data = self.h5_data[sorted_indices]

        # Convert the reordered data to a tensor
        batch_tensor = [torch.tensor(sorted_data, dtype=self.data_dtype)]

        # Reorder h5_ids to match the shuffled order
        h5_ids = self.h5_ids[sorted_indices]
        batch_tensor.append(h5_ids)

        if self.with_output:
            output = []
            for id in h5_ids:
                output_list = list(
                    self.metadata.loc[self.metadata[self.exam_id_col] == id][
                        self.output_cols
                    ].values[0]
                )
                output.append(output_list)
            batch_tensor.append(torch.tensor(output, dtype=self.output_dtype))

        if self.with_metadata:
            metadata = pd.DataFrame()
            for id in h5_ids:
                metadata = pd.concat(
                    [metadata, self.metadata.loc[self.metadata[self.exam_id_col] == id]]
                )
            batch_tensor.append(metadata)

        self.current_batch += 1

        return batch_tensor


class LoadData:
    def __init__(
        self,
        hdf5_path,
        metadata_path,
        code_15_metadata_path,
        hdf5_test_path,
        metadata_test_path,
        batch_size,
        exam_id_col,
        patient_id_col,
        tracing_col,
        output_col,
        tracing_dataset_name,
        exam_id_dataset_name,
        val_size,
        dev_size,
        random_seed,
        data_dtype,
        output_dtype,
        use_fake_data,
        fake_h5_path,
        fake_csv_path,
        use_superclasses,
        block_classes,
        rhythm_classes,
        with_dev,
        norm_metadata,
        cols_to_norm,
    ):
        if use_fake_data:
            hdf5_path = fake_h5_path
            metadata_path = fake_csv_path

        self.hdf5_path = hdf5_path
        self.hdf5_test_path = hdf5_test_path

        self.batch_size = batch_size
        self.current_iter = 0

        self.exam_id_col = exam_id_col
        self.patient_id_col = patient_id_col
        self.tracing_col = tracing_col
        self.output_col = output_col
        self.with_output = output_col is not None

        self.tracing_dataset_name = tracing_dataset_name
        self.exam_id_dataset_name = exam_id_dataset_name

        self.metadata_path = metadata_path
        self.metadata_test_path = metadata_test_path
        self.code_15_metadata = pd.read_csv(code_15_metadata_path)
        self.metadata = pd.read_csv(
            self.metadata_path,
        )
        self.metadata['age'] = self.metadata['age'].round().astype(int)
        for col in self.output_col:
            if col not in self.metadata.columns:
                self.metadata[col] = False

        self.with_dev = with_dev

        self.test_metadata = pd.read_csv(self.metadata_test_path)

        self.hdf5_file = h5py.File(hdf5_path, "r")
        self.exam_id_dataset = self.hdf5_file[self.exam_id_dataset_name]
        self.tracing_dataset = self.hdf5_file[self.tracing_dataset_name]

        self.test_hdf5_file = h5py.File(hdf5_test_path, "r")
        self.test_exam_id_dataset = self.test_hdf5_file[self.exam_id_dataset_name]
        self.test_tracing_dataset = self.test_hdf5_file[self.tracing_dataset_name]

        self.val_size = val_size
        self.dev_size = dev_size if self.with_dev else 0

        self.random_seed = random_seed
        #self.random_seed = random.randint(1, 50)

        self.data_dtype = data_dtype
        self.output_dtype = output_dtype

        self.norm_metadata = norm_metadata
        self.cols_to_norm = cols_to_norm

        self.scaler = MinMaxScaler()

        self.dataset_size = self.metadata.shape[0]

        self.train_set_size = int(
            self.dataset_size
            * (1 - self.val_size - (self.dev_size if self.with_dev else 0))
        )
        self.val_set_size = int(self.dataset_size * self.val_size)
        self.dev_set_size = (
            int(self.dataset_size * self.dev_size) if self.with_dev else 0
        )

        self.train_metadata = None
        self.val_metadata = None
        self.dev_metadata = None

        self.use_superclasses = use_superclasses
        self.block_classes = block_classes
        self.rhythm_classes = rhythm_classes

        self.split_metadata()

    def get_train_set_size(self, batched=False):
        return (
            int(np.ceil(self.train_set_size / self.batch_size))
            if batched
            else self.train_set_size
        )

    def get_val_set_size(self, batched=False):
        return (
            int(np.ceil(self.val_set_size / self.batch_size))
            if batched
            else self.val_set_size
        )

    def apply_norm(self):
        self.scaler.fit(self.train_metadata[self.cols_to_norm])

        self.train_metadata[self.cols_to_norm] = self.scaler.transform(
            self.train_metadata[self.cols_to_norm]
        )
        self.val_metadata[self.cols_to_norm] = self.scaler.transform(
            self.val_metadata[self.cols_to_norm]
        )

        if self.with_dev:
            self.test_metadata[self.cols_to_norm] = self.scaler.transform(
                self.test_metadata[self.cols_to_norm]
            )

    def split_metadata(self):
        self.metadata["block_class"] = self.metadata[self.block_classes].any(axis=1)
        self.metadata["rhythm_class"] = self.metadata[self.rhythm_classes].any(axis=1)
        self.metadata["normal_class"] = ~self.metadata[self.output_col].any(axis=1)

        if self.use_superclasses:
            self.output_col = ["block_class", "rhythm_class", "normal_class"]

        patient_ids = self.metadata[self.patient_id_col].unique()

        np.random.seed(self.random_seed)
        np.random.shuffle(patient_ids)

        if self.dev_size == 0.15:
            print('--> COMO FOI SETADO DEV SIZE=15%, ESTAMOS USANDO O CODE15 COMPLETO COMO DEV SET')
            # When dev_size is 0.15, we have a fixed dev set (code_15_metadata)
            self.dev_metadata = self.code_15_metadata

            # Remaining metadata for train and val split
            remaining_metadata = self.metadata[
                ~self.metadata[self.exam_id_col].isin(self.dev_metadata[self.exam_id_col])
            ]

            remaining_patient_ids = remaining_metadata[self.patient_id_col].unique()
            np.random.shuffle(remaining_patient_ids)

            num_train = int(len(remaining_patient_ids) * (1 - self.val_size))
            num_val = int(len(remaining_patient_ids) * self.val_size)

            self.train_ids = set(remaining_patient_ids[:num_train])
            self.val_ids = set(remaining_patient_ids[num_train:num_train + num_val])

            self.train_metadata = remaining_metadata.loc[
                remaining_metadata[self.patient_id_col].isin(self.train_ids)
            ].reset_index(drop=True)

            self.val_metadata = remaining_metadata.loc[
                remaining_metadata[self.patient_id_col].isin(self.val_ids)
            ].reset_index(drop=True)

        else:
            # Original split logic
            num_train = int(len(patient_ids) * (1 - self.dev_size - self.val_size))
            num_val = int(len(patient_ids) * self.val_size)

            self.train_ids = set(patient_ids[:num_train])
            self.val_ids = set(patient_ids[num_train:num_train + num_val])

            self.train_metadata = self.metadata.loc[
                self.metadata[self.patient_id_col].isin(self.train_ids)
            ].reset_index(drop=True)

            self.val_metadata = self.metadata.loc[
                self.metadata[self.patient_id_col].isin(self.val_ids)
            ].reset_index(drop=True)

            if self.with_dev:
                self.dev_ids = set(patient_ids[num_train + num_val:])

                self.dev_metadata = self.metadata.loc[
                    self.metadata[self.patient_id_col].isin(self.dev_ids)
                ].reset_index(drop=True)

        self.check_dataleakage()


    def get_dataloader(
        self,
        metadata,
        h5_dataset,
        exam_id_dataset,
        balance,
        with_metadata,
        n_workers,
        worker_idx,
        data_frac,
    ):
        metadata = metadata.sample(
            frac=data_frac,
            replace=False,
            random_state=self.random_seed,
            ignore_index=True,
        )

        if balance:
            min_samples = np.min(
                [
                    metadata[metadata[col].astype(int) == 1].shape[0]
                    for col in self.output_col
                ]
            )

            balanced_metadata = pd.DataFrame()

            for col in self.output_col:
                col_metadata = metadata[metadata[col].astype(int) == 1].sample(
                    n=min_samples, replace=False
                )

                balanced_metadata = pd.concat([balanced_metadata, col_metadata])

            metadata = balanced_metadata

        data_indices = np.where(
            np.isin(
                exam_id_dataset[:],
                metadata[self.exam_id_col],
            )
        )[0]

        return MyDataloader(
            h5_data=h5_dataset,
            h5_ids=exam_id_dataset,
            h5_indices=data_indices,
            metadata=metadata,
            exam_id_col=self.exam_id_col,
            output_cols=self.output_col,
            batch_size=self.batch_size,
            data_dtype=self.data_dtype,
            output_dtype=self.output_dtype,
            world_size=n_workers,
            rank=worker_idx,
            with_metadata=with_metadata,
        )

    def check_dataleakage(self):
        train_ids = set(self.train_metadata[self.exam_id_col].unique())
        val_ids = set(self.val_metadata[self.exam_id_col].unique())

        # Check for intersection between any two sets of IDs
        assert (
            len(train_ids.intersection(val_ids)) == 0
        ), "Some IDs are present in both train and validation sets."

    def get_train_dataloader(
        self,
        balance=False,
        with_metadata=False,
        n_workers=1,
        worker_idx=0,
        data_frac=1,
    ):
        return self.get_dataloader(
            self.train_metadata,
            self.tracing_dataset,
            self.exam_id_dataset,
            balance,
            with_metadata,
            n_workers,
            worker_idx,
            data_frac,
        )

    def get_val_dataloader(
        self,
        balance=False,
        with_metadata=False,
        n_workers=1,
        worker_idx=0,
        data_frac=1,
    ):
        return self.get_dataloader(
            self.val_metadata,
            self.tracing_dataset,
            self.exam_id_dataset,
            balance,
            with_metadata,
            n_workers,
            worker_idx,
            data_frac,
        )

    def get_dev_dataloader(
        self,
        balance=False,
        with_metadata=False,
        n_workers=1,
        worker_idx=0,
        data_frac=1,
    ):
        return self.get_dataloader(
            self.dev_metadata,
            self.tracing_dataset,
            self.exam_id_dataset,
            balance,
            with_metadata,
            n_workers,
            worker_idx,
            data_frac,
        )

    def get_test_dataloader(
        self,
        balance=False,
        with_metadata=False,
        n_workers=1,
        worker_idx=0,
        data_frac=1,
    ):
        return (
            self.get_dataloader(
                self.test_metadata,
                self.test_tracing_dataset,
                self.test_exam_id_dataset,
                balance,
                with_metadata,
                n_workers,
                worker_idx,
                data_frac,
            )
            if self.with_dev
            else None
        )
