from device import Devices

def calculate(comp, comm, num_micro_batch):
    return (sum(comp)+sum(comm)+(num_micro_batch-1)*max(max(comp), max(comm)))/1000.0

def calculate2(comp_f, comm_f, comp_b, comm_b, num_device, num_micro_batch):
    warmup = min(2, num_micro_batch)
    t1 = sum(comp_f)+sum(comm_f)+(warmup-1)*max(max(comp_f), max(comm_f))
    t2 = (num_micro_batch-warmup)*(comp_f[-1]+comp_b[-1])
    t3 = sum(comp_b)+sum(comm_b)+(warmup-1)*max(max(comp_b), max(comm_b))
    return (t1+t2+t3)/1000.0

def get_log(filename):
    with open(filename, "r") as f:
        data = f.read().split("\n")[:-1]
   
    real = data[-1].split(":")[-1]
    data = data[:-1]
    l = int((len(data))/2)
    data.reverse()
    backward_comp=[]
    forward_comp = []
    for i in range(l):
        backward_comp.append(float(data[2*i].split(":")[-1]))
        forward_comp.append(float(data[2*i+1].split(":")[-1]))

    return forward_comp, backward_comp, real


if __name__ == '__main__':
    num_device = 16
    num_micro_batch = int(input("num_micro_batch:"))
    filename = input("filename:") 
    forward_comp, backward_comp, real = get_log(filename)
    forward_comp = [int(i*1000000) for i in forward_comp]
    backward_comp = [int(i*1000000) for i in backward_comp]
    forward_comm=[3*1000000]*8+[1.5*1000000]*7
    backward_comm=forward_comm
    #forward_comm=[54]*7+[10*1000000]+[54]*7
    #backward_comm=[30]*7+[10*1000000]+[30]*7
    print(forward_comp)
    print(backward_comp)
    devices = Devices(num_device, num_micro_batch, forward_comp, backward_comp, forward_comm, backward_comm)
    t = devices.run()
    print("simulated estimated time:"+str(t/1000000))
    print(calculate2(forward_comp, forward_comm, backward_comp, backward_comm, num_device, num_micro_batch))
 #   print(calculate(forward_comp, forward_comm, num_micro_batch)+calculate(backward_comp, backward_comm, num_micro_batch))
    print("REAL:"+str(real))
