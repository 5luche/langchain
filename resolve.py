import requests
import json
import os
BASE_URL=  "http://10.189.0.30:9234/getDetailByRequestId?requestId="

mapping={
    "BUSI": ["/Users/liuzhenhua/Documents/GIT/物流综合服务平台/UAT/e6-ms-tms-busi2","master-20260512"],
    "BASE": ["/Users/liuzhenhua/Documents/GIT/物流综合服务平台/UAT/e6-ms-tms-financial","master-20260512"],
    "FINANCIAL": ["/Users/liuzhenhua/Documents/GIT/物流综合服务平台/UAT/e6-ms-tms-base","master-20260512"]
}

def get_error(requestId):
    # 根据requestId获取堆栈信息
    url=BASE_URL+requestId
    response=requests.get(url)
    data=response.json()
    serviceName=data.get("data").get("服务名称")
    fileAndGit=mapping.get(serviceName)
    if fileAndGit==None :
        return "未找到对应的服务"
    file=fileAndGit[0]
    git=fileAndGit[1]
    # 获取最新的代码并切换到最新分支
    os.system("cd "+file+" && git checkout "+git+" && git pull")

    # 将堆栈信息和代码位置交给大模型

if __name__ == '__main__':
    # requestId=input("请输入requestId:")
    requestId="c9e65181d009dfd6"
    print(get_error(requestId))



