import datetime
import time
import numpy as np
import torch
import copy

class ModelInspector():
    """A class for running the model inference with metrics testing. User can
    call the the method to run and test the model and return the tested
    latency and throughput.
    Args:
        batch_num: the number of batches you want to run
        batch_size: batch size you want
        repeat_data: data unit to repeat.
        asynchronous: running asynchronously, default is False.
        sla: SLA, default is 1 sec.
        percentile: The SLA percentile. Default is 95.
    """

    def __init__(
            self,
            repeat_data,
            batch_num,
            batch_size,
            label,
            model,
            name,
            asynchronous: bool = False,
            percentile: int = 95,
            sla: float = 1.0,
    ):
        self.throughput_list = []
        self.latencies = []

        self.asynchronous = asynchronous
        self.percentile = percentile
        self.sla = sla

        self.batch_num = batch_num
        self.batch_size = batch_size
        self.name=name
        self.raw_data = repeat_data
        self.processed_data = self.data_preprocess(self.raw_data)
        self.batches = self.__client_batch_request()
        self.labels=open(label).readlines()
        self.labels = [x.strip() for x in self.labels]
        self.model=model

    def data_preprocess(self, x):
        """Handle raw data, after preprocessing we can get the processed_data, which is using for benchmarking."""
        return x

    def set_batch_size(self, new_bs):
        """update the batch size here.
        Args:
            new_bs: new batch size you want to use.
        """
        self.batch_size = new_bs
        self.batches = self.__client_batch_request()

    def __client_batch_request(self):
        """Batching input data according to the specific batch size."""
        batches = []
        for i in range(self.batch_num):
            batch = []
            for j in range(self.batch_size):
                batch.append(self.processed_data)
            batches.append(batch)
        return batches

    def run_model(self):
        """Running the benchmarking for the specific model on the specific server.
        Args:
            server_name (str): the container's name of Docker that serves the Deep Learning model.
            device (str): Device name. E.g.: cpu, cuda, cuda:1.
        """
        # reset the results
        self.throughput_list = []
        self.latencies = []

        # warm-up
        if self.batch_num > 10:
        
            warm_up_batches = self.batches[:10]

            for batch in warm_up_batches:
                self.start_infer_with_time(batch)
        else:
            raise ValueError("Not enough test values, try to make more testing data.")
        a=[]
        b=[]
        pass_start_time = time.time()
        for batch in self.batches:
            if self.asynchronous:
                ReqThread(self.__inference_callback, self.start_infer_with_time, batch).start()
            else:
                a_batch_latency,res = self.start_infer_with_time(batch)
                self.latencies.append(a_batch_latency)
                a_batch_throughput = self.batch_size / a_batch_latency
                self.throughput_list.append(a_batch_throughput)
                a.append(a_batch_latency)
                b.append(a_batch_throughput)
                print(f' latency: {a_batch_latency:.4f} sec throughput: {a_batch_throughput:.4f} req/sec')
                print("\nThe top-5 labels with corresponding scores are:",res)

        while len(self.latencies) != len(self.batches):
            pass

        pass_end_time = time.time()
        all_data_latency = pass_end_time - pass_start_time
        all_data_throughput = (self.batch_size * self.batch_num) / (pass_end_time - pass_start_time)
        custom_percentile = np.percentile(self.latencies, self.percentile)

        return self.print_results(all_data_throughput, all_data_latency, custom_percentile)

    def infoTuple(self,avg,p50,p95,p99):
        return str([avg,p50,p95,p99])
        
    def __inference_callback(self, a_batch_latency):
        """A callback function which handles the results of a asynchronous inference request.
        Args:
            a_batch_latency: The amount of required for the inference request to complete.
        """
        self.latencies.append(a_batch_latency)
        a_batch_throughput = self.batch_size / a_batch_latency
        self.throughput_list.append(a_batch_throughput)
        # print(" latency: {:.4f}".format(a_batch_latency), 'sec', " throughput: {:.4f}".format(a_batch_throughput), ' req/sec')

    def start_infer_with_time(self, batch_input):

        request = batch_input
        start_time = time.time()

        for r in request:
            # if len(d)==len(r):
            #     print("ok\n")
            # else:
            #     print(len(d),"  ",len(r))
            preds=self.model(copy.deepcopy(r))
            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(preds)
            pred_classes = preds.topk(k=5).indices
            # Map the predicted classes to the label names
            pred_class_names = [self.labels[int(i)] for i in pred_classes[0]]            

        start_time = time.time()
        end_time = time.time()
        return end_time - start_time,pred_class_names


    def print_results(self,throughput,latency,custom_percentile):

        percentile_50 = np.percentile(self.latencies, 50)
        percentile_95 = np.percentile(self.latencies, 95)
        percentile_99 = np.percentile(self.latencies, 99)
        complete_time = datetime.datetime.now()

        print('\n')
        print(f'total batches: {len(self.batches)}, batch_size: {self.batch_size}')
        print(f'total latency: {latency} s')
        print(f'total throughput: {throughput} req/sec')
        print(f'50th-percentile latency: {percentile_50} s')
        print(f'95th-percentile latency: {percentile_95} s')
        print(f'99th-percentile latency: {percentile_99} s')
        # print(f'{self.percentile}th-percentile latency: {custom_percentile} s')
        print(f'completed at {complete_time}')

        return {
            'total_batches': len(self.batches),
            'batch_size': self.batch_size,
            'total_latency': latency,
            'total_throughput': throughput,
            'latency': self.infoTuple(avg=latency / len(self.batches), p50=percentile_50, p95=percentile_95, p99=percentile_99),
            'completed_time': complete_time,
        }


