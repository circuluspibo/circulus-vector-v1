import uvicorn
import json

if __name__ == '__main__':    
    ENV = "OPS"
    #conf = json.load(open('config.json', 'r'))
    #CONF = conf[ENV]
    #print(CONF)

    print("CIRCULUS_ENV: " + ENV)        
    uvicorn.run("main:app","0.0.0.0",59021,reload=False)