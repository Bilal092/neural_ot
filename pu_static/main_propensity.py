from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf
import os
import jax 

import run_lib_ss

FLAGS = flags.FLAGS


datasets = ["Abalone", "Banknote", "Breast-w", "Diabetes", "Haberman", "Heart",
                                 "Ionosphere", "Isolet", "Jm1", "Kc1", "Madelon", "Musk", "Segment",
                                 "Semeion", "Sonar", "Spambase_n", "Vehicle", "Waveform", "Wdbc", "Yeast"]

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train, eval")
flags.mark_flags_as_required(["workdir", "config"])
# flags.DEFINE_float("c", 1.0, "subsetting scalar")

def launch(argv):
    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update("jax_traceback_filtering", "off")
  
    # if isinstance(FLAGS.c, float or int):
    #     if FLAGS.c >=1.0:
    #       FLAGS.config.c = FLAGS.c
    #     else:
    #       raise ValueError(f"c must be float or int >=1.0.")
    # else:
    #     raise TypeError(f"c must be float or scalar >= 1.0.")

    if FLAGS.mode == "train":
       num_tests = 10 if not FLAGS.config.train.num_test else FLAGS.config.train.num_test 
       for test_idx in range(0, num_tests):
           FLAGS.config.seed = test_idx
           run_lib_ss.train(FLAGS.config, FLAGS.workdir)
        
    if FLAGS.mode == "eval":
        run_lib_ss.eval(FLAGS.config, FLAGS.workdir)

  
#   if FLAGS.mode == "train":
#     # Create the working directory
#     # Run the training pipeline
#     run_lib_ss.train(FLAGS.config, FLAGS.workdir)
#   elif FLAGS.mode == "eval":
#     # Run the evaluation pipeline
#     run_lib_ss.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
#   elif FLAGS.mode == "fid_stats":
#     # Run the evaluation pipeline
#     run_lib_ss.fid_stats(FLAGS.config, FLAGS.workdir)
#   else:
#     raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(launch)
