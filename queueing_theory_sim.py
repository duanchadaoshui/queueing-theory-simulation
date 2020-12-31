# 定义M/M/K/C模型仿真函数
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt



def queueing_mmkc(lamda,mu,K,C,N):
    '''
    参数说明：
    lamda:到达率，满足参数为lamda的泊松分布
    mu：服务率，满足参数为1/mu的均值为mu的指数分布
    K：服务台数量
    C：队列限制长度（buffer）
    N: 仿真长度（小时）
    return: 一个长度为N的列表，为每时刻个队列长度 queuelength_list
            一个长度为np.random.poisson(lam=lamda,size=1)[0]的列表，为所有顾客的等待时间 waittime_list
            
            
    模拟N个小时，秒级模拟
    
    '''
    queuelength_list = []
    waittime_list = []
    # 入队时的时间
    queue_list = []
    # 服务需要的时间
    sever_list = []
    
    '''
    # N时间内到达个数为arrival_n
    rng = np.random.default_rng()
    arrival_n = rng.poisson(lam=lamda*N,size=1)[0]
    # arrival_n个时间的到达时间为0到N的均匀分布
    S_n = rng.integers(N*60*60,size=arrival_n)
    S_n.sort()
    '''
    rng = np.random.default_rng()
    # 第一个事件的到达时间
    arrival_t = int(rng.exponential(1/lamda,size=1)[0]*60*60)
    # N长度的仿真
    for i in range(0,N*60*60):
        arrival_t -= 1
        if len(sever_list) > 0:
            # 服务队列不空，有需要服务的
            sever_list = (np.array(sever_list)-1).tolist()
            
            while 0 in sever_list:
                sever_list.remove(0)
        
        # 记录等候队列长度
        queuelength_list.append(len(queue_list))
            
        # 等待队列入队     
        while len(sever_list) < K:
            # 服务台有空
            if len(queue_list) <= 0:
                break
            else:
                # 加入服务台
                sever_time = int(rng.exponential(mu,size=1)[0]*60*60)
                while sever_time == 0:
                    sever_time = int(rng.exponential(mu,size=1)[0]*60*60)
                sever_list.append(sever_time)
                
                waittime_list.append(i - queue_list.pop(0)-1)
                
        # 到达事件入队        
        if arrival_t <= 0:
            # 有到达的事件
            if len(queue_list) < C:
                # 等候队列不满,记录入队时间
                queue_list.append(i)
                
                arrival_t = int(rng.exponential(1/lamda,size=1)[0]*60*60)
            else:
                # 等候队列已满，丢包
                waittime_list.append(-1)
                print('丢包')
                arrival_t = int(rng.exponential(1/lamda,size=1)[0]*60*60)
        

    #plt.plot(queuelength_list)
    # plt.show()
    return queuelength_list,waittime_list


lost_num = 0
time_start=time.time()
# result0,result1 = queueing_mmkc(lamda=20,mu=1/24,K=1,C=50000*60*60,N=50000)

re0 = [0 for x in range(0,500)]
re1 = [0 for x in range(0,500)]
for j in range(0,500):
    result0,result1 = queueing_mmkc(lamda=20,mu=1/24,K=1,C=10000*60*60,N=100)
    time_end=time.time()
    # print('totally cost',time_end-time_start)
    re0[j] = np.mean(result0)
    re1[j] = np.mean(result1)
    print(re0[j])

print('queue length = ',np.mean(re0))
while -1 in result1:
    result1.remove(-1)
    lost_num += 1
print('lost numbers = ',lost_num)
print('wait time = ',np.mean(re1))
time_end=time.time()
print('totally cost',time_end-time_start)
