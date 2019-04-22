import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import nbimporter
import numpy as np
import math
import sys
import os
if('..' not in sys.path):
    sys.path.insert(0,'..')
if('../..' not in sys.path):
    sys.path.insert(0,'../..')
from Maze.Maze import Maze
from Maze.MazeGenerator import MazeGenerator
from Agents.Worker import Worker
from Agents.Spider import Spider
from Agents.Queen import Queen
from Main.Simulator import Simulator
from Maths.Cord import Cord
from Maths.Action import Action
#from Maths.DDDQN import DQNSolver
from Maths.DQNSolver import DQNSolver

class GameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
       # print("Initialising")
        m= MazeGenerator(10,10) #"Test,10,10,4,0,4,9,1111211111100000000111100110011000000001100001011111011000111000000001100101010110000000011111311111"
        #"Test,10,10,4,0,4,9,1111211111100000000111100110011000000001100001011111011000111000000001100101010110000000011111311111"
        #"Test,6,6,4,5,1,0,131111100051105001100051150001111121" #"Test,10,10,4,0,4,9,1111211151100000000111100110011000000001100001511111011000111000000001100501050110000000011111311111"
        #"Test,4,4,2,0,2,3,1125100110011531" MazeGenerator() 
        self.maze= Maze(m)
        self.s=Simulator(self.maze)
        self.spanP=5
        self.spanM=5
        self.wNumber=5
        self.qNumber=0
        self.sNumber=0
        self.pList=[]
        self.qList=[]
        self.mList=[]
        self.pPos=[]
        self.qPos=[]
        self.mPos=[]
        self.pStateList=[]
        self.qStateList=[]
        self.mStateList=[]
        self.finishedP=[]
        self.finishedQ=[]
        self.history=m+"|"+"0"
        self.finished=0
        self.count=0
        self.eaten=0
        self.queenEaten=False
        self.queenLeft=False
        
        #rewards
        self.wReward_not_possible=-50
        self.wReward_wall=-50
        self.wReward_entrance=-20
        self.wReward_finished_before=0
        self.wReward_exit=self.maze.height*self.maze.width*20
        self.wReward_towards_exit=-1
        self.wReward_toQueen=-1
        self.wReward_atQueen=self.maze.height*self.maze.width*20
        self.wReward_repeat_pos=-20
        self.wReward_else=-3
        self.wReward_queenEaten=-5000
        
        self.qReward_not_possible=-50
        self.qReward_wall=-50
        self.qReward_entrance=-20
        self.qReward_finished_before=0
        self.qReward_exit=self.maze.height*self.maze.width*20
        self.qReward_towards_exit=-1
        self.qReward_repeat_pos=-20
        self.qReward_else=-3
        
        self.sReward_not_possible=-50
        self.sReward_wall=-50
        self.sReward_eat=1000
        self.sReward_eatQueen=5000
        self.sReward_towards_prey=-1
        self.sReward_repeat_pos=-20
        self.sReward_else=-3
        
        #for book-keeping/need to update with spiders
        self.folderName= "Span_"+str(self.spanP)+"Dim_"+str(self.maze.height)+"_"+str(self.maze.width)
        #m+"_maze_"+str(self.wNumber)+"_workers_"+str(self.spanP)+"_span_"+str(self.wReward_not_possible)+"_not poss_"+str(self.wReward_wall)+"_wall_"+str(self.wReward_entrance)+"_ent_"+str(self.wReward_finished_before) +"_finished already_"+str(self.wReward_exit)+"_exit_"+str(self.wReward_towards_exit)+"_to exit_"+str(self.wReward_repeat_pos) +"_rep pos_"+str(self.wReward_else)+"_else " ####
            
        self.maxIter=10*self.maze.height*self.maze.width
        self.completeStop=False
        
        for j in range(self.wNumber):
            p=Worker(self.maze, self.spanP)
            self.s.add(p)
            self.pList.append(p)
            self.pPos.append(p.getPos())        
            
        for j in range(self.qNumber):
            q=Queen(self.maze)
            self.s.add(q)
            self.qList.append(q)
            self.qPos.append(q.getPos())
            
        #self.history+="|"
        action_space=[]
        
        for j in range(self.sNumber):
            s=Spider(self.maze)
            self.s.add(s)
            self.mList.append(s)
            self.mPos.append(s.getPos())
            
        for j in range(self.wNumber):
            state= np.asarray(p.getAugView(p.getPos(),self.spanP,self.pPos, self.qPos, self.mPos))
            self.pStateList.append(state)
            self.history+="#"+p.getName()+"-"+p.getPos().CordToString()
            
        for j in range(self.qNumber):
            #state= np.asarray(q.getView())
            #self.qStateList.append(state)
            self.history+="#"+q.getName()+"-"+q.getPos().CordToString()
            
        for j in range(self.sNumber):
            state= np.asarray(s.getAugView(s.getPos(),self.spanM,self.pPos, self.qPos, self.mPos))
            self.mStateList.append(state)
            self.history+="#"+s.getName()+"-"+s.getPos().CordToString()  #fix writeup
        self.history+="|"
        action_space=[]
        
        for i in range(0,len(Action)):            
            action_space.append(i)
        self.action_space_worker=np.asarray(action_space)
        self.observation_space_worker= math.pow(2*self.spanP+1,2)
        
        self.action_space_queen=np.asarray(action_space)
        self.observation_space_queen= (self.maze.height+2*self.spanP)*(self.maze.width+2*self.spanP) #math.pow(2*self.spanP+1,2)
        
        self.action_space_spider=np.asarray(action_space)
        self.observation_space_spider= math.pow(2*self.spanM+1,2)
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])
        self.maze.printMaze()
        
        self.dqn_solver_worker = DQNSolver(int(self.observation_space_worker), len(self.action_space_worker))
        self.dqn_solver_queen = DQNSolver(int(self.observation_space_queen), len(self.action_space_queen))
        self.dqn_solver_spider = DQNSolver(int(self.observation_space_spider), len(self.action_space_spider))
        self.pReward=0
        self.qReward=0
        self.mReward=0
        
    def stepAll(self):
        terminal=False
        #walls move
        time=''
        if(len(self.pList)>0):
            time=str(self.pList[0].getTime())
        elif(len(self.qList)>0):
            time=str(self.qList[0].getTime())
        elif(len(self.mList)>0):
            time=str(self.mList[0].getTime())
        #print("TIME",time)
        self.history+=time+"#"+ self.maze.returnAllClearString()
        blocked=[]
        for p in self.pList:
            blocked.append(p.getPos())
        for q in self.qList:
            blocked.append(q.getPos())
        for m in self.mList:
            blocked.append(m.getPos())
                        
        self.maze.WallMove(blocked)
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])            
          
        
        #prey move_workers
        index=0
        for p in self.pList:
            if(p.exploring):
                state = np.reshape(self.pStateList[index],  [1,int(self.observation_space_worker)])
                #print(self.pStateList[index],[1,int(self.observation_space_worker)])
                #print(state)
                action = self.dqn_solver_worker.act(state)
                #print("1")
                state_next, reward, terminal, info = self.step(p, action, index)
                state_next = np.reshape(state_next, [1,int(self.observation_space_worker)])
                self.dqn_solver_worker.remember(state, action, reward, state_next, terminal)
                self.pStateList[index]=state_next
                self.dqn_solver_worker.experience_replay()
                self.pReward+=reward
                self.history+="#"+p.getName()+"-"+p.getPos().CordToString()
                p.routeToQueen.append(p.getPos())
            else:
                # need reward, state_next
                state_next, reward = p.goToQueen( self.qList[0].getPos(), self.spanP, self.pPos, self.qPos, self.mPos, self.wReward_not_possible, self.wReward_wall, self.wReward_toQueen, self.wReward_atQueen, self.wReward_repeat_pos, self.wReward_else)
                
                state_next = np.reshape(state_next, [1,int(self.observation_space_worker)])
                self.pStateList[index]=state_next
                self.pReward+=reward
                self.history+="#"+p.getName()+"-"+p.getPos().CordToString()
            
            #print(p.getName(),"moved to",p.getPos().CordToString(),"at",time, p.exploring)
            self.pPos[index]=p.getPos()    
            if(p.exploring):    
                #p.exploring=not (p.isQueenOnEdge(self.spanP,self.pPos, self.qPos, self.mPos))
                if(p.isQueenOnEdge(self.spanP,self.pPos, self.qPos, self.mPos)):
                    #print("i")
                    p.save_state(self.spanP,self.pPos, self.qPos, self.mPos)
                    #print("ii")
                    print("Ant","Taking Snapshot",p.savedTime)
                #print("iii")
            if((not p.exploring) and (p.getPos().equals(self.qList[0].getPos()))):
                #print(p.savedState,p.savedTime,len(q.history)) 
                #for key in q.history:
                 #   print(key,q.history.get(key).CordToString())
                #print(p.getName(),"returned to the queen ",self.qList[0].history.get(p.savedTime).CordToString(),"at",time)
                p.exploring=True
                #print("Workers at:",[[c.getName(), c.getPos().CordToString()] for c in self.pList])
                
                self.qList[0].combine(p.savedState,p.savedTime)
            index+=1
            
            #print("ANT",p.getName(), p.getTime(), p.getPos().CordToString(), "Snapshot", p.savedTime)
        #self.history+="|" 
        
        #queen moves
        index=0
        for q in self.qList:
            oldPos=q.getPos()
            if(len(q.view)>0):
                view=q.norm_view((self.maze.height+2*self.spanP),(self.maze.width+2*self.spanP))
                state = np.reshape(view,  [1,int(self.observation_space_queen)])
                action = self.dqn_solver_queen.act(state)
                state_next, reward, terminal, info = self.step(q, action, index)
                viewNew=q.norm_view((self.maze.height+2*self.spanP),(self.maze.width+2*self.spanP))
                state_next = np.reshape(viewNew, [1,int(self.observation_space_queen)])
                self.dqn_solver_queen.remember(state, action, reward, state_next, terminal)
                #self.mStateList[index]=state_next
                self.dqn_solver_queen.experience_replay()
                self.qReward+=reward
            else:
                q.TimeStep+=1
            self.history+="#"+q.getName()+"-"+q.getPos().CordToString()
            self.qPos[0]=q.getPos()
            #print("(Global)Queen going",oldPos.CordToString(),"->",q.getPos().CordToString(), "at", q.getTime(),time)
            action=4
            if(q.getPos().Y==oldPos.Y+1):
                action=0
            elif(q.getPos().Y==oldPos.Y-1):
                action=1
            elif(q.getPos().X==oldPos.X+1):
                action=2
            elif(q.getPos().X==oldPos.X-1):
                action=3
            
            q.updateView(action)
            #print("Updated view:")
            #q.show_span(q.getView())
            q.history[q.getTime()]=q.getPos()  
            #print("----------------------------")
            index+=1

        for p1 in self.pList:
            q=p1.getPos()
            if(len(self.qList)>0):
                q=self.qList[0].getPos()
            p1.updateVulnerability(self.pPos,q)
        for q1 in self.qList:
            q1.updateVulnerability(self.pPos)   
        
        #predators move
        index=0
        for s in self.mList:
            state = np.reshape(self.mStateList[index],  [1,int(self.observation_space_spider)])
            action = self.dqn_solver_spider.act(state)
            state_next, reward, terminal, info = self.step(s, action, index)
            state_next = np.reshape(state_next, [1,int(self.observation_space_spider)])
            self.dqn_solver_spider.remember(state, action, reward, state_next, terminal)
            self.mStateList[index]=state_next
            self.dqn_solver_spider.experience_replay()
            self.mReward+=reward
            self.history+="#"+s.getName()+"-"+s.getPos().CordToString()
            self.mPos[index]=s.getPos()
            index+=1
        self.history+="|"
        
        trueTermination=False
        #print("Check ",self.finished,len(self.pList))
        #print(" ")
        self.maxIter-=1
        if( ((self.finished==len(self.pList))and (len(self.qList)==0)) or(self.queenEaten) or (self.queenLeft) or (self.maxIter==0)):            
            path=u"DATA/"+u"Testing/exp2/"+ self.folderName
            if not os.path.exists(path):
                #print(path)
                #print(len(path.encode()))
                os.makedirs(path)
            file=open(path+u"/GamesData.txt","a+")
            file.write(self.history+"\n")
            file.close()
            terminal=True
            trueTermination=True
        
        
        #if(self.maxIter==0):
        #    terminal=True
       #     trueTermination=False
        #print(" ")    
        return self.pReward, self.mReward, terminal, trueTermination, self.eaten

    def step(self, agent, action, index): 
        reward=0
        terminal=False
        info={}
        
        oldPosition=agent.getPos()
        state_Next=np.empty(1)      
        
        if(str(type(agent).__name__)=="Worker"):          
            if(self.action_space_worker[action] in self.maze.WhichWayIsClear(oldPosition, True)):
                agent.Do(self.action_space_worker[action],self.maze)
                state_Next=np.asarray(agent.getAugView(agent.getPos(),self.spanP,self.pPos, self.qPos, self.mPos))         
             
                reward+=agent.getReward(agent.getPos(), True,oldPosition,agent.getAugView(agent.getPos(),self.spanP,self.pPos, self.qPos, self.mPos), self.wReward_not_possible, self.wReward_wall, self.wReward_entrance, self.wReward_finished_before, self.wReward_exit, self.wReward_towards_exit, self.wReward_repeat_pos, self.wReward_else)
                self.count+=1      
        
            else:              
                agent.Do(self.action_space_worker[4],self.maze)
                state_Next=np.asarray(agent.getAugView(agent.getPos(),self.spanP,self.pPos, self.qPos, self.mPos))             
            
                reward+=agent.getReward(agent.getPos(), False,oldPosition,agent.getAugView(agent.getPos(),self.spanP,self.pPos, self.qPos, self.mPos), self.wReward_not_possible, self.wReward_wall, self.wReward_entrance, self.wReward_finished_before, self.wReward_exit, self.wReward_towards_exit, self.wReward_repeat_pos, self.wReward_else)
                  
            
            if(self.maze.CheckExit(agent.getPos()) and agent not in self.finishedP):
                self.finished+=1
                self.finishedP.append(agent)     
                
        elif(str(type(agent).__name__)=="Spider"): 
            if(self.action_space_spider[action] in self.maze.WhichWayIsClear(oldPosition, True)):
                agent.Do(self.action_space_spider[action],self.maze)
                state_Next=np.asarray(agent.getAugView(agent.getPos(),self.spanM,self.pPos, self.qPos, self.mPos))                      
                  
                reward+=agent.getReward(agent.getPos(), True,oldPosition,agent.getAugView(agent.getPos(),self.spanM,self.pPos, self.qPos, self.mPos), self.sReward_not_possible, self.sReward_wall,[p.getPos() for p in self.pList], self.sReward_eat, self.sReward_towards_prey, self.sReward_repeat_pos, self.sReward_else)
                if((agent.getPos().equals(self.qList[0].getPos())) and (self.qList[0].vulnerable)):
                    reward+=self.sReward_eatQueen
                    self.pReward+=self.wReward_queenEaten
                    self.qList=[]
                    self.queenEaten=True
                    terminal=True
                remove=[]
                for prey in self.pList:
                    if((agent.getPos().equals(prey.getPos())) and (prey.vulnerable)):
                        remove.append(prey)
                        self.eaten+=1
                        self.pReward-=self.wReward_exit
                for corpse in remove:
                    self.pList.remove(corpse)
                        
                
            else:  
                agent.Do(self.action_space_spider[4],self.maze)
                state_Next=np.asarray(agent.getAugView(agent.getPos(),self.spanM,self.pPos, self.qPos, self.mPos))             
            
                reward+=agent.getReward(agent.getPos(), False,oldPosition,agent.getAugView(agent.getPos(),self.spanM,self.pPos, self.qPos, self.mPos), self.sReward_not_possible, self.sReward_wall,[p.getPos() for p in self.pList], self.sReward_eat, self.sReward_towards_prey, self.sReward_repeat_pos, self.sReward_else)
                
        elif(str(type(agent).__name__)=="Queen"): 
            if(self.action_space_queen[action] in self.maze.WhichWayIsClear(oldPosition, True)):
                agent.Do(self.action_space_queen[action],self.maze)
                state_Next=np.asarray(agent.getView())                     
                 
                #self, pos, possible, oldPos, view, rNotPos, rWall, rEnt, rFinBef, rEx, rToEx, rRep, rElse    
                reward+=agent.getReward(agent.getPos(), True,oldPosition,agent.getView(), self.qReward_not_possible, self.qReward_wall, self.qReward_entrance, self.qReward_finished_before, self.qReward_exit, self.qReward_towards_exit, self.qReward_repeat_pos, self.qReward_else)
                
            else:  
                agent.Do(self.action_space_queen[4],self.maze)
                state_Next=np.asarray(agent.getView())             
            
                reward+=agent.getReward(agent.getPos(), False,oldPosition,agent.getView(), self.qReward_not_possible, self.qReward_wall, self.qReward_entrance, self.qReward_finished_before, self.qReward_exit, self.qReward_towards_exit, self.qReward_repeat_pos, self.qReward_else)
                
            if(self.maze.CheckExit(agent.getPos()) and agent not in self.finishedQ):
                self.finishedQ.append(agent) 
                self.queenLeft=True
        
        #print(agent.getTime(), agent.getName(), oldPosition.CordToString(), " -> ",agent.getPos().CordToString())
        return state_Next, reward, terminal, info
    
    def reset(self):        
        #print("Resetting")
        self.maze= Maze(self.maze.mazeString)
        self.pList=[]
        self.qList=[]
        self.mList=[]
        self.pPos=[]
        self.qPos=[]
        self.mPos=[]
        self.pStateList=[]
        self.mStateList=[]
        self.qStateList=[]
        self.finishedP=[]
        self.finishedQ=[]
        self.history=self.maze.mazeString+"|"+"0"
        self.finished=0
        self.finishedP=[]
        self.count=0        
        self.queenLeft=False
        self.maxIter=10*self.maze.height*self.maze.width
        
        
        for j in range(self.wNumber):
            p=Worker(self.maze,self.spanP)
            self.pList.append(p) 
                
        for k in range(self.qNumber):
            q=Queen(self.maze)
            self.qList.append(q) 
                
        for h in range(self.sNumber):
            s=Spider(self.maze)
            self.mList.append(s) 
                
        #print(len(self.pList),len(self.qList),len(self.mList))
        
        self.eaten=0
        self.queenEaten=False
        
        for p in self.pList:
            p.setInitPos(Cord(self.maze.getInitialX(),self.maze.getInitialY()))
            self.pPos.append(p.getPos())        
        
        for q in self.qList:
            q.setInitPos(Cord(self.maze.getInitialX(),self.maze.getInitialY()))
            self.qPos.append(q.getPos())
                    
        for q in self.mList:
            q.setInitPos(Cord(q.start.X,q.start.Y))
            self.mPos.append(q.getPos())
            
        for p in self.pList:
            state=np.asarray(p.getAugView(p.getPos(),self.spanP,self.pPos, self.qPos, self.mPos))
            self.pStateList.append(state)
            self.history+="#"+p.getName()+"-"+p.getPos().CordToString()
        
        for q in self.qList:
            #state=np.asarray(p.getAugView(p.getPos(),self.spanP,self.pPos, self.qPos, self.mPos))
            #self.pStateList.append(state)
            self.history+="#"+q.getName()+"-"+q.getPos().CordToString()
            
        for q in self.mList:
            state=np.asarray(q.getAugView(q.getPos(),self.spanM,self.pPos, self.qPos, self.mPos))
            self.mStateList.append(state)
            self.history+="#"+q.getName()+"-"+q.getPos().CordToString()
            
        self.history+="|"
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])
        self.pReward=0
        self.qReward=0
        self.mReward=0
        #print(self.maze.mazeString)
        #self.maze.printMaze()
        return self.pStateList
    
    def resetNewMaze(self): 
        mazesizeh=self.maze.height+10
        mazesizew=self.maze.width+10
        if(self.maze.height==5):
            mazesizeh=10
            mazesizew=10
        m= MazeGenerator(mazesizeh,mazesizew)
        #"Test,10,10,4,0,4,9,1111211111100000000111100110011000000001100001011111011000111000000001100101010110000000011111311111" #MazeGenerator()  
        self.maze= Maze(m)
        self.s=Simulator(self.maze)
        self.pList=[]
        self.qList=[]
        self.mList=[]
        self.pPos=[]
        self.qPos=[]
        self.mPos=[]
        self.pStateList=[]
        self.qStateList=[]
        self.mStateList=[]
        self.history=m+"|"+"0"
        self.finished=0
        self.finishedP=[]
        self.finishedQ=[]
        self.count=0
        self.eaten=0
        self.queenEaten=False
        self.queenLeft=False
        
         #rewards
        self.wReward_not_possible=-50
        self.wReward_wall=-50
        self.wReward_entrance=-20
        self.wReward_finished_before=0
        self.wReward_exit=self.maze.height*self.maze.width*20
        self.wReward_towards_exit=-1
        self.wReward_toQueen=-1
        self.wReward_atQueen=self.maze.height*self.maze.width*20
        self.wReward_repeat_pos=-20
        self.wReward_else=-3
        
        self.qReward_not_possible=-50
        self.qReward_wall=-50
        self.qReward_entrance=-20
        self.qReward_finished_before=0
        self.qReward_exit=self.maze.height*self.maze.width*20
        self.qReward_towards_exit=-1
        self.qReward_repeat_pos=-20
        self.qReward_else=-3
        
        self.sReward_not_possible=-50
        self.sReward_wall=-50
        self.sReward_eat=1000
        self.sReward_eatQueen=5000
        self.sReward_towards_prey=-1
        self.sReward_repeat_pos=-20
        self.sReward_else=-3
        
        #for book-keeping
        #self.wNumber=self.wNumber+1
        self.folderName= "Span_"+str(self.spanP)+"Dim_"+str(self.maze.height)+"_"+str(self.maze.width) #m+"_maze_"+str(self.wNumber)+"_workers_"+str(self.spanP)+"_span_"+str(self.wReward_not_possible)+"_not poss_"+str(self.wReward_wall)+"_wall_"+str(self.wReward_entrance)+"_ent_"+str(self.wReward_finished_before) +"_finished already_"+str(self.wReward_exit)+"_exit_"+str(self.wReward_towards_exit)+"_to exit_"+str(self.wReward_repeat_pos) +"_rep pos_"+str(self.wReward_else)+"_else "
            
        self.maxIter=10*self.maze.height*self.maze.width           
        self.completeStop=False
        if(self.wReward_exit==0):
            self.completeStop=True
            
        
        for j in range(self.wNumber):
            p=Worker(self.maze, self.spanP)
            self.s.add(p)
            self.pList.append(p)
            self.pPos.append(p.getPos())        
            
        for j in range(self.qNumber):
            q=Queen(self.maze)
            self.s.add(q)
            self.qList.append(q)
            self.qPos.append(q.getPos())
            
        #self.history+="|"
        action_space=[]
        
        for j in range(self.sNumber):
            s=Spider(self.maze)
            self.s.add(s)
            self.mList.append(s)
            self.mPos.append(s.getPos())
            
        for j in range(self.wNumber):
            state= np.asarray(p.getAugView(p.getPos(),self.spanP,self.pPos, self.qPos, self.mPos))
            self.pStateList.append(state)
            self.history+="#"+p.getName()+"-"+p.getPos().CordToString()
            
        for j in range(self.qNumber):
            #state= np.asarray(q.getView())
            #self.qStateList.append(state)
            self.history+="#"+q.getName()+"-"+q.getPos().CordToString()
            
        for j in range(self.sNumber):
            state= np.asarray(s.getAugView(s.getPos(),self.spanM,self.pPos, self.qPos, self.mPos))
            self.mStateList.append(state)
            self.history+="#"+s.getName()+"-"+s.getPos().CordToString()  #fix writeup
                    
        self.history+="|"
            
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])
        self.maze.printMaze()
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])
        
        self.dqn_solver_worker = DQNSolver(int(self.observation_space_worker), len(self.action_space_worker))
        self.dqn_solver_queen = DQNSolver(int(self.observation_space_queen), len(self.action_space_queen))
        self.dqn_solver_spider = DQNSolver(int(self.observation_space_spider), len(self.action_space_spider))
        self.pReward=0
        self.qReward=0
        self.mReward=0
        
        return self.pStateList
        
    def render(self, mode='human', close=False):
        self.s.display()