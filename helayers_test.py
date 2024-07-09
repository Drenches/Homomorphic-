import pyhelayers
import os
import sys
import h5py
import numpy as np
import pdb
import time

print('Imported pyhelayers',pyhelayers.VERSION)
hyper_params = pyhelayers.PlainModelHyperParams()
nnp = pyhelayers.NeuralNetPlain()
pdb.set_trace()

#initialize the NN architecture and weights from the json and h5 files that we stored before.
output_dir = 'example_path'
nnp.init_from_files(hyper_params, [os.path.join(output_dir, "model.json"), os.path.join(output_dir, "model.h5")])

pred_batch_size=16

he_run_req = pyhelayers.HeRunRequirements()
he_run_req.set_he_context_options([pyhelayers.DefaultContext()])
he_run_req.optimize_for_batch_size(pred_batch_size)

# Set the requirements and run the model
profile = pyhelayers.HeModel.compile(nnp, he_run_req)
print(profile)
context = pyhelayers.HeModel.create_context(profile)

nn = pyhelayers.NeuralNet(context)
nn.encode_encrypt(nnp, profile)

with h5py.File(os.path.join(output_dir, "x_test.h5")) as f:
    x_test = np.array(f["x_test"])
with h5py.File(os.path.join(output_dir, "y_test.h5")) as f:
    y_test = np.array(f["y_test"])


def extract_batch(x_test, y_test, batch_size, batch_num):
    num_samples = x_test.shape[0]
    num_lebels = y_test.shape[0]

     # assert same size
    assert(num_samples == num_lebels)

    # calc start and end index
    start_index = batch_num * batch_size
    if start_index >= num_samples:
        raise RuntimeError('Not enough samples for batch number ' +
                           str(batch_num) + ' when batch size is ' + str(batch_size))
    end_index = min(start_index + batch_size, num_samples)

    batch_x = x_test.take(indices=range(start_index, end_index), axis=0)
    batch_y = y_test.take(indices=range(start_index, end_index), axis=0)

    return (batch_x, batch_y)

plain_samples, labels = extract_batch(x_test, y_test, pred_batch_size, 0)
print('Batch of size',pred_batch_size,'loaded')

iop = nn.create_io_processor()
samples = pyhelayers.EncryptedData(context)
iop.encode_encrypt_inputs_for_predict(samples,[plain_samples])
print('Test data encrypted')

startTime = time.time()

predictions = pyhelayers.EncryptedData(context)
nn.predict(predictions,samples)

latency    = round(time.time() - startTime,5)
amortized_latency = round(latency/pred_batch_size,5)
print(f"Latency = {latency} seconds, Amortized latency={amortized_latency} seconds")

plain_predictions = iop.decrypt_decode_output(predictions)
expected_pred=nnp.predict([plain_samples])
diff=expected_pred-plain_predictions
print('L2 distance between HE and plain predictions',np.linalg.norm(diff))