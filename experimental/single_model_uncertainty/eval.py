# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Basic eval functions for Uncertainty Baselines."""

import os.path
import time
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

from absl import logging
import edward2 as ed
import robustness_metrics as rm
import tensorflow.compat.v2 as tf
import uncertainty_baselines as ub
from tensorboard.plugins.hparams import api as hp

_EVAL_SLEEP_SECS = 5

_TensorDict = Dict[str, tf.Tensor]
EvalStepFn = Callable[[Iterator[_TensorDict]], _TensorDict]

_EvalSetupResult = Tuple[
    EvalStepFn,
    tf.data.Dataset,
    tf.summary.SummaryWriter,
    Optional[EvalStepFn],
    Optional[tf.data.Dataset],
    Optional[tf.summary.SummaryWriter],
    Optional[EvalStepFn],
    Optional[tf.data.Dataset],
    Optional[tf.summary.SummaryWriter]]


def eval_step_fn(model: tf.keras.Model,
                 strategy: tf.distribute.Strategy,
                 metrics: Dict[str, Union[tf.keras.metrics.Metric,
                                          rm.metrics.KerasMetric]],
                 iterations_per_loop: int,
                 label_key: str = 'labels',
                 mean_field_factor: float = -1) -> EvalStepFn:
  """Generator for a function to run iters_per_loop validation/test steps."""

  @tf.function
  def eval_step(train_iterator: Iterator[_TensorDict]) -> _TensorDict:
    def step(per_replica_inputs: _TensorDict) -> None:
      """The function defining a single validation/test step."""
      features = per_replica_inputs['features']
      labels = per_replica_inputs[label_key]
      logits = model(features, training=False)
      if isinstance(logits, (tuple, list)):
        logits, covmat = logits
      else:
        per_core_batch_size, _ = logits.get_shape().as_list()
        covmat = tf.eye(per_core_batch_size)

      logits = ed.layers.utils.mean_field_logits(
          logits, covmat, mean_field_factor=mean_field_factor)

      predictions = tf.nn.softmax(logits, axis=-1)
      if label_key != 'labels':
        predictions = tf.reduce_max(predictions, axis=-1)
      # Later when metric.result() is called, it will return the computed
      # result, averaged across replicas.
      for metric in metrics.values():
        if isinstance(metric, tf.keras.metrics.Metric):
          metric.update_state(labels, predictions)  # pytype: disable=attribute-error
        else:
          metric.add_batch(predictions, label=labels)
      return

    for metric in metrics.values():
      metric.reset_states()
    for _ in tf.range(iterations_per_loop):  # Note the use of tf.range.
      ub.utils.call_step_fn(strategy, step, next(train_iterator))
    total_results = {name: value.result() for name, value in metrics.items()}
    # Metrics from Robustness Metrics (like ECE) will return a dict with a
    # single key/value, instead of a scalar.
    total_results = {
        k: (list(v.values())[0] if isinstance(v, dict) else v)
        for k, v in total_results.items()
    }
    return total_results

  return eval_step


def run_eval_epoch(
    current_step: int,
    test_fn: EvalStepFn,
    test_dataset: tf.data.Dataset,
    test_summary_writer: tf.summary.SummaryWriter,
    val_fn: Optional[EvalStepFn] = None,
    val_dataset: Optional[tf.data.Dataset] = None,
    val_summary_writer: Optional[tf.summary.SummaryWriter] = None,
    ood_fn: Optional[EvalStepFn] = None,
    ood_dataset: Optional[tf.data.Dataset] = None,
    ood_summary_writer: Optional[tf.summary.SummaryWriter] = None,
    hparams: Optional[Dict[str, Any]] = None):
  """Run one evaluation epoch on the test and optionally validation splits."""
  val_outputs_np = None
  if val_dataset:
    val_iterator = iter(val_dataset)
    val_outputs = val_fn(val_iterator)
    with val_summary_writer.as_default():  # pytype: disable=attribute-error
      if hparams:
        hp.hparams(hparams)
      for name, metric in val_outputs.items():
        tf.summary.scalar(name, metric, step=current_step)
    val_outputs_np = {k: v.numpy() for k, v in val_outputs.items()}
    logging.info(
        'Validation metrics for step %d: %s', current_step, val_outputs_np)
  if ood_dataset:
    ood_iterator = iter(ood_dataset)
    ood_outputs = ood_fn(ood_iterator)
    with ood_summary_writer.as_default():  # pytype: disable=attribute-error
      if hparams:
        hp.hparams(hparams)
      for name, metric in ood_outputs.items():
        tf.summary.scalar(name, metric, step=current_step)
    ood_outputs_np = {k: v.numpy() for k, v in ood_outputs.items()}
    logging.info(
        'OOD metrics for step %d: %s', current_step, ood_outputs_np)

  test_iterator = iter(test_dataset)
  test_outputs = test_fn(test_iterator)
  with test_summary_writer.as_default():
    if hparams:
      hp.hparams(hparams)
    for name, metric in test_outputs.items():
      tf.summary.scalar(name, metric, step=current_step)
  return val_outputs_np, {k: v.numpy() for k, v in test_outputs.items()}


def setup_eval(validation_dataset_builder: Optional[ub.datasets.BaseDataset],
               test_dataset_builder: ub.datasets.BaseDataset,
               batch_size: int,
               strategy,
               trial_dir: str,
               model: tf.keras.Model,
               metrics: Dict[str, Union[tf.keras.metrics.Metric,
                                        rm.metrics.KerasMetric]],
               ood_dataset_builder: Optional[ub.datasets.BaseDataset] = None,
               ood_metrics: Optional[Dict[str, tf.keras.metrics.Metric]] = None,
               mean_field_factor: float = -1) -> _EvalSetupResult:
  """Setup the test and optionally validation loggers, step fns and datasets."""
  test_dataset = test_dataset_builder.load(batch_size=batch_size)
  test_dataset = strategy.experimental_distribute_dataset(test_dataset)
  test_summary_writer = tf.summary.create_file_writer(
      os.path.join(trial_dir, 'test'))
  num_test_steps = test_dataset_builder.num_examples // batch_size
  test_fn = eval_step_fn(
      model,
      strategy,
      metrics,
      iterations_per_loop=num_test_steps,
      mean_field_factor=mean_field_factor)
  ood_fn = None
  ood_dataset = None
  ood_summary_writer = None

  if ((ood_dataset_builder and not ood_metrics) or
      (not ood_dataset_builder and ood_metrics)):
    raise ValueError('Both ood_dataset_builder and ood_metrics must be'
                     ' specified.')
  if ood_dataset_builder:
    ood_dataset = ood_dataset_builder.load(batch_size=batch_size)
    ood_dataset = strategy.experimental_distribute_dataset(ood_dataset)
    ood_summary_writer = tf.summary.create_file_writer(
        os.path.join(trial_dir, 'ood'))
    num_ood_steps = ood_dataset_builder.num_examples // batch_size

    ood_fn = eval_step_fn(
        model,
        strategy,
        ood_metrics,
        iterations_per_loop=num_ood_steps,
        label_key='is_in_distribution',
        mean_field_factor=mean_field_factor)

  # Have to have separate val_fn and test_fn because otherwise tf.function
  # retraces the function each time, which is very slow, because we are passing
  # in a Python dict of metrics and int for iterations_per_loop.
  val_fn = None
  val_dataset = None
  val_summary_writer = None
  if validation_dataset_builder:
    num_val_steps = validation_dataset_builder.num_examples // batch_size
    val_dataset = validation_dataset_builder.load(batch_size=batch_size)
    val_dataset = strategy.experimental_distribute_dataset(val_dataset)
    val_summary_writer = tf.summary.create_file_writer(
        os.path.join(trial_dir, 'validation'))
    if num_val_steps == num_test_steps:
      val_fn = test_fn
    else:
      # The metrics are reset at the start of each call to {val,test}_fn, so
      # reusing them is safe.
      val_fn = eval_step_fn(
          model,
          strategy,
          metrics,
          iterations_per_loop=num_val_steps,
          mean_field_factor=mean_field_factor)

  return (
      test_fn, test_dataset, test_summary_writer,
      val_fn, val_dataset, val_summary_writer,
      ood_fn, ood_dataset, ood_summary_writer)


def run_eval_loop(validation_dataset_builder: Optional[ub.datasets.BaseDataset],
                  test_dataset_builder: ub.datasets.BaseDataset,
                  batch_size: int,
                  model: tf.keras.Model,
                  trial_dir: str,
                  train_steps: int,
                  strategy: tf.distribute.Strategy,
                  metrics: Dict[str, Union[tf.keras.metrics.Metric,
                                           rm.metrics.KerasMetric]],
                  checkpoint_step: int = -1,
                  hparams: Optional[Dict[str, Any]] = None,
                  ood_dataset_builder: Optional[ub.datasets.BaseDataset] = None,
                  ood_metrics: Optional[Dict[str,
                                             tf.keras.metrics.Metric]] = None,
                  mean_field_factor: float = -1):
  """Evaluate the model on the validation and test splits and record metrics."""
  (test_fn, test_dataset, test_summary_writer, val_fn, val_dataset,
   val_summary_writer, ood_fn, ood_dataset, ood_summary_writer) = setup_eval(
       validation_dataset_builder, test_dataset_builder, batch_size, strategy,
       trial_dir, model, metrics, ood_dataset_builder, ood_metrics,
       mean_field_factor)

  checkpoint = tf.train.Checkpoint(model=model)
  last_eval_step = -1
  # Note that this will only grab the latest checkpoint, so if multiple
  # checkpoints are saved while this is sleeping, it will skip the ones in
  # between.
  while True:
    # Check for a new checkpoint, and if there is not one, sleep for several
    # seconds.
    if checkpoint_step >= 0:
      checkpoint_path = os.path.join(
          trial_dir, 'ckpt-{}'.format(checkpoint_step))
    else:
      checkpoint_path = tf.train.latest_checkpoint(trial_dir)
    if not checkpoint_path:
      last_checkpoint_step = last_eval_step
    else:
      last_checkpoint_step = int(checkpoint_path.split('-')[-1])
    if last_checkpoint_step == last_eval_step:
      logging.info(
          'No new checkpoints since step %d (latest path is %s). Sleeping '
          'for %d seconds...',
          last_eval_step,
          checkpoint_path,
          _EVAL_SLEEP_SECS)
      time.sleep(_EVAL_SLEEP_SECS)
      continue

    # Restore from the latest checkpoint and evalutate on the validation and
    # test splits.
    last_eval_step = last_checkpoint_step
    logging.info('Restoring model from checkpoint %s.', checkpoint_path)
    checkpoint.restore(checkpoint_path).assert_existing_objects_matched()
    # Only write hparams on the final step.
    written_hparams = None
    if last_eval_step >= train_steps:
      written_hparams = hparams
    run_eval_epoch(
        last_eval_step,
        test_fn,
        test_dataset,
        test_summary_writer,
        val_fn,
        val_dataset,
        val_summary_writer,
        ood_fn,
        ood_dataset,
        ood_summary_writer,
        hparams=written_hparams,
        )
    if last_eval_step >= train_steps:
      break
