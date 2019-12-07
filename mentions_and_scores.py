from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import util

if __name__ == "__main__":
  config = util.initialize_from_env()
  log_dir = config["log_dir"]

  # Input file in .jsonlines format.
  input_filename = sys.argv[2]

  # Predictions will be written to this file in .jsonlines format.
  output_filename = sys.argv[4]

  config["top_span_ratio"] = float(sys.argv[3])

  model = util.get_model(config)
  saver = tf.train.Saver()

  with tf.Session() as session:
    model.restore(session)

  
    with open(output_filename, "w") as output_file:
      with open(input_filename) as input_file:
        for example_num, line in enumerate(input_file.readlines()):
          example = json.loads(line)
          tensorized_example = model.tensorize_example(example, is_training=False)
          feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
          starts, ends, candidate_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
          score_map = {}
          for (start, end, score) in zip(starts, ends, candidate_scores):

            # Only for BERT ones
            if "subtoken_map" in example:
              token_start = example["subtoken_map"][start]
              token_end = example["subtoken_map"][end]
            else:
              token_start = start
              token_end = end

            score_map[(token_start, token_end)] = score

          output_obj = {
            example["doc_key"] : [
                {
                    "start": token_start,
                    "end": token_end,}for (token_start, token_end), score in score_map.items()]
          }

          output_file.write(json.dumps(output_obj) + "\n")

