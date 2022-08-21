from collections import deque
from math import inf


## the input numbers of this program need to be rounded to integers.

class Devices(object):
    def __init__(self, num_device, num_micro_batch, device_forward_comp_time, device_backward_comp_time, device_forward_comm_time, device_backward_comm_time):
        self.num_device = num_device
        self.num_stage = num_device # currently assuming num_stage = num_device.
        self.device_forward_comp_time = list(device_forward_comp_time)
        self.device_backward_comp_time = list(device_backward_comp_time)
        self.device_forward_comm_time = list(device_forward_comm_time)
        self.device_backward_comm_time = list(device_backward_comm_time)
        assert len(device_forward_comp_time) == num_device
        assert len(device_forward_comm_time) == num_device - 1
        assert len(device_backward_comp_time) == num_device
        assert len(device_backward_comm_time) == num_device - 1
        self.workload_forward_comp = dict()
        self.workload_forward_comm = dict()
        self.workload_backward_comp = dict()
        self.workload_backward_comm = dict()
        self.timer = 0
        self.num_micro_batch = num_micro_batch
        self.warmup_batch_num = [0 for i in range(num_device)]
        self.remaining_batch_num = [0 for i in range(num_device)]
        self.f = []
        self._1f1b_end = False
        self.processed_num_of_device = list()
        self.processed_num_of_device_backward = list()
        for i in range(num_device):
            self.processed_num_of_device_backward.append(0)
            self.workload_forward_comp[i] = deque()
            self.workload_forward_comm[i] = deque()
            self.workload_backward_comp[i] = deque()
            self.workload_backward_comm[i] = deque()
            self.f.append(True)
            self.processed_num_of_device.append(0)
            self.warmup_batch_num[i] = min(self.num_device - i + 1, self.num_micro_batch)
            self.remaining_batch_num[i] = num_micro_batch - self.warmup_batch_num[i]

    def initialize(self):
        assert self.warmup_batch_num[0] > 0
        self.workload_forward_comp[0].append(self.device_forward_comp_time[0])

    def nearest_finish_time(self):
        time = inf
        for i in range(self.num_device - 1):
            if self.workload_forward_comm[i]:
                if self.workload_forward_comm[i][0] < time:
                    time = self.workload_forward_comm[i][0]
            if  self.workload_backward_comm[i]:
                if self.workload_backward_comm[i][0] < time:
                    time = self.workload_backward_comm[i][0]

        for i in range(self.num_device):
            if self.f[i] and self.workload_forward_comp[i]:
                if self.workload_forward_comp[i][0] < time:
                    time = self.workload_forward_comp[i][0]
            if self.f[i]==False and self.workload_backward_comp[i]:
                if self.workload_backward_comp[i][0] < time:
                    time = self.workload_backward_comp[i][0]

        return time

    def warmup_step(self):
        time = self.nearest_finish_time()
        self.timer += time
        for i in range(self.num_device):
            if self.workload_forward_comp[i] and self.f[i]:
                self.workload_forward_comp[i][0] -= time
            if self.workload_backward_comp[i]:
                self.workload_backward_comp[i][0] -= time
            if i < self.num_device - 1:
                if self.workload_forward_comm[i]:
                    self.workload_forward_comm[i][0] -= time
                if self.workload_backward_comm[i]:
                    self.workload_backward_comm[i][0] -= time

        for i in range(self.num_device):
            if self.workload_forward_comp[i] and self.workload_forward_comp[i][0] == 0:
                self.workload_forward_comp[i].popleft()
                self.processed_num_of_device[i] += 1
                if i == 0:
                    self.workload_forward_comm[0].append(self.device_forward_comm_time[0])
                    if self.processed_num_of_device[0] <= self.warmup_batch_num[0] and self.processed_num_of_device[0]<self.num_micro_batch:
                        self.workload_forward_comp[0].append(self.device_forward_comp_time[0])
                else:
                    if i < self.num_device:
                        self.workload_forward_comm[i].append(self.device_forward_comm_time[0])
            if i < self.num_device - 1 and self.workload_forward_comm[i] and self.workload_forward_comm[i][0] == 0:
                self.workload_forward_comm[i].popleft()
                self.workload_forward_comp[i+1].append(self.device_forward_comp_time[i+1])

        if self.processed_num_of_device[self.num_device-1]==self.warmup_batch_num[self.num_device-1]+1:
            self.workload_backward_comp[self.num_device-1].append(self.device_backward_comp_time[self.num_device-1])
        self.update_fb_state()
        self.print_status(time)

    def update_fb_state(self):
        for i in range(self.num_device):
            if self.warmup_batch_num[i] < self.num_micro_batch:
        #        print("device_"+str(i)+", processed_"+str(self.processed_num_of_device[i])+", warmup_"+str(self.warmup_batch_num[i]))
                assert self.processed_num_of_device[i] <= self.warmup_batch_num[i]+1
                if self.processed_num_of_device[i] <= self.warmup_batch_num[i]:
                    self.f[i]=True
                else:
                    self.f[i]=False
            else:
                # warmup_batch_num[i] == num_micro_batch
                assert self.processed_num_of_device[i] <= self.warmup_batch_num[i]
                if self.processed_num_of_device[i] < self.warmup_batch_num[i]:
                    self.f[i]=True
                else:
                    self.f[i]=False
                

    def _1f1b_start(self):
        if self.processed_num_of_device[self.num_device-1]==self.warmup_batch_num[self.num_device-1]+1:
            return True
        return False

    def warmup(self):
        while self._1f1b_start()==False:
            self.warmup_step()

    def print_init_status(self):
        print("NUM DEVICES: "+str(self.num_device))
        print("FORWARD_COMPUTE_TIME "+str(self.device_forward_comp_time))
        print("FORWARD_COMMUNICATE_TIME"+str(self.device_forward_comm_time))
        print("BACKWARD_COMPUTE_TIME"+str(self.device_backward_comp_time))
        print("BACKWARD_COMMUNICATE_TIME"+str(self.device_backward_comm_time))
        print("NUM MICRO BATCH: "+str(self.num_micro_batch))
        print("\n")
        self.print_helper()

    def print_helper(self):
        for i in range(self.num_device):
            workload_comp_f = ""
            workload_comp_b = ""
            workload_comm_f = ""
            workload_comm_b = ""
            if self.workload_forward_comp[i]:
                workload_comp_f += "f:"+str(",".join(tuple([str(i) for i in self.workload_forward_comp[i]])))
            else:
                workload_comp_f += "f:    "
            if self.workload_backward_comp[i]:
                workload_comp_b += "b:"+str(",".join(tuple([str(i) for i in self.workload_backward_comp[i]])))
            else:
                workload_comp_b += "b:    "
            if i < self.num_device -1 and self.workload_forward_comm[i]:
                workload_comm_f += "f:"+str(",".join(tuple([str(i) for i in self.workload_forward_comm[i]])))
            else:
                workload_comm_f += "f:    "
            if i < self.num_device -1 and self.workload_backward_comm[i]:
                workload_comm_b += "b:"+str(",".join(list([str(i) for i in self.workload_backward_comm[i]])))
            else:
                workload_comm_b += "b:    "
            forb="F" if self.f[i] else "B"
            print("DEVICE "+str(i)+forb+" processed_forward "+str(self.processed_num_of_device[i])+" processed_backward "+str(self.processed_num_of_device_backward[i]))
            print("       "+str(workload_comp_f)+"  "+str(workload_comp_b))
            if i < self.num_device-1:
                print("       "+str(workload_comm_f)+"  "+workload_comm_b)
        print("\n")

    def print_status(self, time):
        print("#################  AFTER "+str(time)+" secs, TIME CONSUMED: "+str(self.timer))
        print("\n")
        self.print_helper()

    def do_1f1b(self):
        while self._1f1b_end==False: # set to false when backward_processed_num_of_device_n==num_micro_batch
            self.do_1f1b_step()

    def do_1f1b_step(self):
        time = self.nearest_finish_time()
        self.timer += time
        for i in range(self.num_device):
            if self.f[i]:
                if self.workload_forward_comp[i]:
                    self.workload_forward_comp[i][0] -= time
            else:
                if self.workload_backward_comp[i]:
                    self.workload_backward_comp[i][0] -= time
            if i < self.num_device -1:
                if self.workload_forward_comm[i]:
                    self.workload_forward_comm[i][0] -= time
                if self.workload_backward_comm[i]:
                    self.workload_backward_comm[i][0] -= time
        
        for i in range(self.num_device):
            if self.f[i]:
                if self.workload_forward_comp[i]:
                    if self.workload_forward_comp[i][0] == 0:
                        self.f[i]=False
                        self.workload_forward_comp[i].popleft()
                        self.processed_num_of_device[i]+=1
                        if i < self.num_device-1:
                            self.workload_forward_comm[i].append(self.device_forward_comm_time[i])
                        if i == self.num_device -1:
                            self.workload_backward_comp[i].append(self.device_backward_comp_time[i])
                        if i == 0 and self.processed_num_of_device[0]<self.warmup_batch_num[0]+1 and self.processed_num_of_device[0] < self.num_micro_batch:
                            self.workload_forward_comp[i].append(self.device_forward_comp_time[0])
                        if self.processed_num_of_device[i] < self.warmup_batch_num[i]+1:
                            self.f[i]=True
            else:
                if self.workload_backward_comp[i]:
                    if self.workload_backward_comp[i][0] == 0:
                        self.processed_num_of_device_backward[i] += 1
                        self.f[i]=True
                        if self.processed_num_of_device[i] == self.num_micro_batch:
                            self.f[i]=False
                        self.workload_backward_comp[i].popleft()
                        if i > 0:
                            self.workload_backward_comm[i-1].append(self.device_backward_comm_time[i-1])
                        if i == self.num_device -1 :
                            if self.processed_num_of_device_backward[i] == self.remaining_batch_num[self.num_device-1]:
                                self._1f1b_end = True
                        if i == 0 and self.processed_num_of_device[0] < self.num_micro_batch:
                            self.workload_forward_comp[0].append(self.device_forward_comp_time[0])

            if i < self.num_device -1 and self.workload_forward_comm[i]:
                if self.workload_forward_comm[i][0] == 0:
                    self.workload_forward_comm[i].popleft()
                    self.workload_forward_comp[i+1].append(self.device_forward_comp_time[i+1])
            if i < self.num_device -1 and self.workload_backward_comm[i]:
                if self.workload_backward_comm[i][0] == 0:
                    self.workload_backward_comm[i].popleft()
                    self.workload_backward_comp[i].append(self.device_backward_comp_time[i])

        self.print_status(time)

    def cooldown(self):
        n = self.warmup_batch_num[self.num_device -1]
        t = sum(self.device_backward_comp_time)+sum(self.device_backward_comm_time)+(n-1)*max(max(self.device_backward_comp_time), max(self.device_backward_comm_time))

        return t
    def run(self):
        self.initialize()
        self.print_init_status()
        self.warmup()
        
        print("WARMUP FINISHED")

        self.do_1f1b()
        time = self.cooldown()
        print("FINISHED!!")
        return self.timer + time 
