from modifications.gmm import gmm
import torch

def search(corpus, tags, dump, property_projection, classes = (True, False)):
    # (corpus, tags) should be related by tags = tag(corpus, ...)
    # from tag.py. property_projection should be a function from
    # lists of target tags to a finite set of classes (by default True, False).

    tag_dict = {
        tuple(a): [
            property_projection(*t)
            for t in b
        ] for a, b in zip(corpus, tags)
    }

    def balanced_accuracy(pred, gold):
        n_classes = len(classes)

        # Pred, gold are going to be tokens x neurons of predicted
        # class indices. We create (n_classes) dimensions on dimension 0
        # and compare to arange to get the predicted sets for each class.
        tokens, neurons = pred.shape
        predicted_positives = pred.unsqueeze(0)\
            .expand(n_classes, tokens, neurons)\
            .eq(torch.arange(n_classes).unsqueeze(1)\
            .unsqueeze(2).long()).float()
        gold_positives = gold.unsqueeze(0)\
            .expand(n_classes, tokens)\
            .eq(torch.arange(n_classes).unsqueeze(1).long())\
            .unsqueeze(2).expand_as(predicted_positives).float()

        # For each class, get number of correct positives
        correct_positives = predicted_positives * gold_positives

        # ... and proportion of correct positives
        correct_ratio = correct_positives.sum(1) / gold_positives.sum(1)

        # Mean proportion of correct positives over classes
        return correct_ratio.mean(0)


    # We run gmm to search for predictive neurons of the given classes.
    results, parameters = gmm(
        source = corpus,
        dump = dump,
        manual_tag = lambda line: tag_dict[tuple(line)],
        tags = classes,
        simulate_balanced = True,
        scoring_function = balanced_accuracy
    )

    # Find the best neuron and its means, stdevs per class
    (layer, neuron), index, score = results[0]

    means, stdevs = parameters

    # These will each be vectors of length (classes)
    mean, stdev = means[:, index], stdevs[:, index]

    return (layer, neuron), (mean, stdev)
