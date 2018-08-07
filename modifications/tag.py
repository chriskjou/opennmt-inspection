from tqdm import tqdm

def tag_aligned(
        source=[],
        target=[],
        alignments=[],
        source_tagger=lambda x: [None for _ in x],
        target_tagger=lambda x: [None for _ in x]):

    # We will assemble a list of
    # pairs (source_tags, [target_tags]) where target_tags
    # is a list (more properly interpreted as a multiset)
    # of tags this was aligned to.
    result = []

    # Iterate through and tag
    for source_line, target_line, alignment in tqdm(zip(source, target, alignments), total=len(source)):

        # Tag
        source_tags = source_tagger(source_line)
        target_tags = target_tagger(target_line)

        line_result = list(zip(source_tags, [[] for _ in source_tags]))

        for source_location, target_location in alignment:
            line_result[source_location][1].append(
                target_tags[target_location])

        line_result = [(a, sorted(b)) for a, b in line_result]

        result.append(line_result)

    return result
