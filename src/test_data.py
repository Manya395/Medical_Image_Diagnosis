from src.data_loader import get_data_generators

train_gen, val_gen, test_gen = get_data_generators()

print("Train batches:", len(train_gen))
print("Validation batches:", len(val_gen))
print("Test batches:", len(test_gen))

x, y = next(train_gen)
print("Batch image shape:", x.shape)
print("Batch label shape:", y.shape)