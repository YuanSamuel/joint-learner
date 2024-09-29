from collections import Counter

def count_labels(dataloader):
    label_counter = Counter()

    # Loop through the batches in the dataloader
    for _, data in enumerate(dataloader):
        # Assume that the third element of the batch is the label
        # Modify based on the actual output of your `__getitem__` method
        _, labels = data

        labels = labels.flatten().tolist()
        
        # Update the counter with the batch's labels
        label_counter.update(labels)
    
    # print(label_counter)

    return label_counter[1], label_counter[0]