import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import nbimporter
import numpy as np
import math
import sys
if('..' not in sys.path):
    sys.path.insert(0,'..')
if('../..' not in sys.path):
    sys.path.insert(0,'../..')
from Maze.Maze import Maze
from Maze.MazeGenerator import MazeGenerator
from Agents.Worker import Worker
from Main.Simulator import Simulator
from Maths.Cord import Cord
from Maths.Action import Action

class GameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
       # print("Initialising")
        m= "Test,6,6,4,5,1,0,131111100011101001100011110001111121" #"Test,10,10,4,0,4,9,1111211111100000000111100110011000000001100001011111011000111000000001100101010110000000011111311111"
        #"Test,6,6,4,5,1,0,131111100051105001100051150001111121" #"Test,10,10,4,0,4,9,1111211151100000000111100110011000000001100001511111011000111000000001100501050110000000011111311111"
        #"Test,4,4,2,0,2,3,1125100110011531" MazeGenerator() 
        self.maze= Maze(m)
        self.s=Simulator(self.maze)
        self.span=6
        self.number=2
        self.pList=[]
        self.stateList=[]
        self.finishedP=[]
        self.history=m+"|"+"0"
        self.finished=0
        self.count=0
        for j in range(self.number):
            p=Worker(self.maze)
            self.s.add(p)
            self.pList.append(p)
            state= np.asarray(p.getView(p.getPos(),self.span))
            self.stateList.append(state)
            self.history+="#"+p.getName()+"-"+p.getPos().CordToString()
        self.history+="|"
        action_space=[]
        
        for i in range(0,len(Action)):            
            action_space.append(i)
        import itertools
        possible_actions=[action_space]*self.number
        print(action_space)
        self.action_space=np.asarray(list(itertools.product(*possible_actions)))
        #print(self.action_space)
        #self.action_space=np.asarray(action_space)
        self.observation_space= math.pow(2*self.span+1,2)*self.number
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])
        self.maze.printMaze()
        
        

    def step(self, action):  
        stateList=self.stateList
        state_nextList=np.empty([1,2*self.span+1,2*self.span+1])
        reward=0
        terminal=False
        info={}
        
        wallMove=False
        index=0
        
        possible=0
        #print(" ")
        for q in self.pList: 
            oldPosition=q.getPos()
            #print("Checking POssible",self.action_space[action][index], self.maze.WhichWayIsClear(oldPosition, True))
            if(self.action_space[action][index] in self.maze.WhichWayIsClear(oldPosition, True)):                
                possible+=1
            index+=1    
        #print("Possible ",possible)        
        index=0
        for p in self.pList:            
            oldPosition=p.getPos()
            state_next=np.empty(1)
            
            #print(p.getName(), oldPosition.CordToString(),self.action_space[action][index],self.maze.WhichWayIsClear(oldPosition, True))
            
            if(self.action_space[action][index] in self.maze.WhichWayIsClear(oldPosition, True) and possible==self.number):
                p.Do(self.action_space[action][index],self.maze)
                #print(p.getName(), oldPosition.CordToString(), " to ",p.getPos().CordToString(),Action(self.action_space[action][index]))#, "Moving",self.maze.returnMoving())
                state_Next=np.asarray(p.getView(p.getPos(),self.span))
                reward+=p.getReward(p.getPos(), True,oldPosition,p.getView(p.getPos(),self.span))
                wallMove=True
                self.count+=1            
            else:                
                state_Next=np.asarray(p.getView(p.getPos(),self.span))                
                reward+=p.getReward(p.getPos(), False,oldPosition,p.getView(p.getPos(),self.span))
            
            state_nextList=np.append(state_nextList,[state_Next], axis=0)
            index+=1
            if(self.maze.CheckExit(p.getPos()) and p not in self.finishedP):
                self.finished+=1
                self.finishedP.append(p)
        state_nextList=np.delete(state_nextList,0,axis=0)
        
        
        if(wallMove):
            self.history+=str(self.pList[0].getTime())+"#"+self.maze.returnAllClearString()
            blocked=[]
            for p in self.pList:
                blocked.append(p.getPos())
                self.history+="#"+p.getName()+"-"+p.getPos().CordToString()            
            self.maze.WallMove(blocked)
            self.shortestRoute=len(self.maze.GetOptimalRoute()[0])            
            self.history+="|"   
            
        if(self.finished==len(self.pList)):
            file=open("GamesData.txt","a+")
            file.write(self.history+"\n")
            file.close()
            terminal=True            
        
        return state_nextList, reward, terminal, info
    
    def reset(self):        
        #print("Resetting")
        self.maze= Maze(self.maze.mazeString)
        self.stateList=[]
        self.history=self.maze.mazeString+"|"+"0"
        self.finished=0
        self.finishedP=[]
        self.count=0
        for p in self.pList:
            p.setInitPos(Cord(self.maze.getInitialX(),self.maze.getInitialY()))
            state=np.asarray(p.getView(p.getPos(),self.span))
            self.stateList.append(state)
            self.history+="#"+p.getName()+"-"+p.getPos().CordToString()
        self.history+="|"
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])
        #print(self.maze.mazeString)
        #self.maze.printMaze()
        return self.stateList
    
    def resetNewMaze(self):        
        m= MazeGenerator()  
        self.maze= Maze(m)
        self.s=Simulator(self.maze)
        self.pList=[]
        self.stateList=[]
        self.history=m+"|"+"0"
        self.finished=0
        self.finishedP=[]
        self.count=0
        for j in range(self.number):
            p=Worker(self.maze)
            self.pList.append(p)
            self.s.add(p)
            state= np.asarray(p.getView(p.getPos(),self.span))
            self.stateList.append(state)
            self.history+="#"+p.getName()+"-"+p.getPos().CordToString()
        self.history+="|"
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])
        self.maze.printMaze()
        self.shortestRoute=len(self.maze.GetOptimalRoute()[0])
        return self.stateList
        
    def render(self, mode='human', close=False):
        self.s.display()