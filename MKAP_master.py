from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import time
import math
import random

class MKAP:
	def __init__(self,m,n,r,inst,InstType=None):
		self.m = m
		self.n = n
		self.r = r
		self.inst = inst
		self.p = []
		self.w = []
		self.c = []
		self.type = InstType
		self.Sk = {k:[] for k in range(self.r)}
		self.InstName = None

	def ReadSet1(self):
		self.InstName = 'probT1_10e3_'+self.type+'_R50_T'+"{:03d}".format(self.r)+'_M'+"{:03d}".format(self.m)+'_N'+"{:04d}".format(self.n)+'_seed'+"{:02d}".format(self.inst)
		f = open('MKAP_instances/SET1/'+self.InstName+'.inp','r')
		#f = open("../../DWdecomp/MKAP_instances/SET1/probT1_10e3_"+self.type+"_R50_T"+"{:03d}".format(self.r)+"_M"+"{:03d}".format(self.m)+"_N"+"{:04d}".format(self.n)+"_seed"+"{:02d}".format(self.inst)+".inp",'r')
		x = f.readlines()
		f.close()
		lineinfo = x[3].strip(' \n').split()
		for i in range(self.m):
			self.c.append(int(lineinfo[i]))
		for j in range(self.n):
			lineinfo = x[4+j].strip('\n').split()
			self.p.append(int(lineinfo[1]))
			self.w.append(int(lineinfo[2]))
			self.Sk[int(lineinfo[3])].append(j)

	def ReadSet2(self):
		self.InstName = 'probT2_10e3_'+self.type+'_R50_T'+"{:03d}".format(self.r)+'_M'+"{:03d}".format(self.m)+'_N'+"{:04d}".format(self.n)+'_seed'+"{:02d}".format(self.inst)
		f = open('MKAP_instances/SET2/'+self.InstName+'.inp','r')
		x = f.readlines()
		f.close()
		lineinfo = x[3].strip(' \n').split()
		for i in range(self.m):
			self.c.append(int(lineinfo[i]))
		for j in range(self.n):
			lineinfo = x[4+j].strip('\n').split()
			self.p.append(int(lineinfo[1]))
			self.w.append(int(lineinfo[2]))
			self.Sk[int(lineinfo[3])].append(j)

	def ReadNew(self):
		self.InstName = 'new'+self.type+'_R'+str(self.r)+'_M'+str(self.m)+'_N'+str(self.n)+'_seed'+str(self.inst)
		f = open('MKAP_instances/NEW/'+self.InstName+'.txt','r')
		x = f.readlines()
		f.close()
		lineinfo = x[0].strip(' \n').split()
		for j in range(self.n):
			self.w.append(int(lineinfo[j]))
		lineinfo = x[1].strip(' \n').split()
		for j in range(self.n):
			self.p.append(int(lineinfo[j]))
		lineinfo = x[2].strip(' \n').split()
		for i in range(self.m):
			self.c.append(int(lineinfo[i]))
		self.Sk = {k:[k*int(self.n/self.r)+p for p in range(int(self.n/self.r))] for k in range(self.r)}

	def CompareDWLevel(self):
		Sk = self.Sk
		# Calculate root bound
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.update()
		m_LP = m.relax()
		m_LP.update()
		m_LP.optimize()
		LB_LP = m_LP.ObjBound
		m.params.NodeLimit = 1
		m.update()
		t0 = time.time()
		m.optimize()
		RootTime0 = time.time()-t0
		Root0 = m.ObjBound
		del m
		del m_LP

		t0_Dual = time.time()
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.params.SolutionLimit = 2
		m.update()
		m.optimize()
		UB = m.objval
		del m

		DualMaster = Model()
		DualMaster.modelSense = GRB.MAXIMIZE
		DualMaster.params.OutputFlag = 0
		DualMaster.params.threads = 4
		lam = DualMaster.addVars(self.n,lb=-float('inf'),ub=0)
		mu = DualMaster.addVars(self.m,lb=-float('inf'),ub=0)
		# theta <= V(lambda)
		theta = DualMaster.addVars(self.m,self.r,lb=-float('inf'))
		DualObj = DualMaster.addVar(lb=-float('inf'),ub=UB)
		DualMaster.addConstr(DualObj == quicksum(lam[j] for j in range(self.n))+quicksum(mu[i] for i in range(self.m))+quicksum(theta[i,k] for i in range(self.m) for k in range(self.r)))
		DualMaster.setObjective(DualObj)
		subIP = {}
		subx = {}
		suby = {}
		for i in range(self.m):
			for k in range(self.r):
				subIP[i,k] = Model()
				subIP[i,k].params.OutputFlag = 0
				subIP[i,k].params.threads = 4
				subx[i,k] = subIP[i,k].addVars(Sk[k],vtype=GRB.BINARY)
				suby[i,k] = subIP[i,k].addVar(vtype=GRB.BINARY)
				subIP[i,k].addConstr(quicksum(self.w[j]*subx[i,k][j] for j in Sk[k]) <= self.c[i]*suby[i,k])
		
		lam_soln = {}
		mu_soln = {}
		LB = -float('inf')
		cutplBounds = []
		levelBounds = []

		DWCuts = {(i,k):np.empty((0,len(Sk[k])+2)) for i in range(self.m) for k in range(self.r)}
		for DualMethod in ["cutpl","level"]:
			iterN = 0
			StopCondt = False
			t0 = time.time()
			while StopCondt == False:
				iterN += 1
				print(iterN,LB,UB,time.time()-t0)
				DualMaster.update()
				DualMaster.optimize()
				UB = DualMaster.objval
				lam_old = lam_soln.copy()
				mu_old = mu_soln.copy()
				if DualMethod == 'cutpl' or (DualMethod == 'level' and iterN == 1):
					for j in range(self.n):
						lam_soln[j] = lam[j].x
					for i in range(self.m):
						mu_soln[i] = mu[i].x
				elif DualMethod == 'level':
					lt = UB-0.3*(UB-LB)
					levelCon = DualMaster.addConstr(DualObj >= lt)
					DualMaster.setObjective(-quicksum((lam[j]-lam_old[j])*(lam[j]-lam_old[j]) for j in range(self.n))-quicksum((mu[i]-mu_old[i])*(mu[i]-mu_old[i]) for i in range(self.m)))
					DualMaster.optimize()
					if DualMaster.status == 2:
						for j in range(self.n):
							lam_soln[j] = lam[j].x
						for i in range(self.m):
							mu_soln[i] = mu[i].x
					else:
						print('QP solver having numerical issues...')
						break
					DualMaster.remove(levelCon)
					DualMaster.setObjective(DualObj)
				if UB-LB < 1e-6*(min(abs(UB),abs(LB))+1) or iterN >= 1000:
					StopCondt = True
					print('Dual problem terminates!')

				# Solve "pricing" subproblem
				for i in range(self.m):
					for k in range(self.r):
						subIP[i,k].setObjective(-quicksum((self.p[j]+lam_soln[j])*subx[i,k][j] for j in Sk[k])-mu_soln[i]*suby[i,k])
						subIP[i,k].update()
						subIP[i,k].optimize()
						if subIP[i,k].status == 2:
							subx_soln = {j:subx[i,k][j].x for j in Sk[k]}
							suby_soln = suby[i,k].x
							DualMaster.addConstr(theta[i,k] <= -sum(self.p[j]*subx_soln[j] for j in Sk[k])-quicksum(subx_soln[j]*lam[j] for j in Sk[k])-suby_soln*mu[i])
							DWCuts[i,k] = np.append(DWCuts[i,k],np.array([[-self.p[j]-lam_soln[j] for j in Sk[k]]+[-mu_soln[i],-subIP[i,k].objval]]),axis=0)
						else:
							print('subIP error!')
				LB_new = sum(lam_soln[j] for j in range(self.n))+sum(mu_soln[i] for i in range(self.m))+sum(subIP[i,k].objval for i in range(self.m) for k in range(self.r))
				if LB_new > LB:
					LB = LB_new
				if DualMethod == "cutpl":
					cutplBounds.append((iterN,time.time()-t0,LB,UB))
				else:
					levelBounds.append((iterN,time.time()-t0,LB,UB))

		for DualMethod in ["cutpl","level"]:
			WrtStr = ''
			if DualMethod == "cutpl":
				for tp in cutplBounds:
					WrtStr += str(tp)+'\n'
			else:
				for tp in levelBounds:
					WrtStr += str(tp)+'\n'

			f = open('LogFiles/DWLevel/'+self.InstName+'_'+DualMethod+'.txt','a')
			f.write(WrtStr)
			f.close()





	def ExtCompare(self,tLimit = 10*60):
		#Sk = {k:[k*int(self.n/self.r)+p for p in range(int(self.n/self.r))] for k in range(self.r)}
		Sk = self.Sk
		# Calculate root bound
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.update()
		m_LP = m.relax()
		m_LP.update()
		m_LP.optimize()
		LB_LP = m_LP.ObjBound
		m.params.NodeLimit = 1
		m.update()
		t0 = time.time()
		m.optimize()
		RootTime0 = time.time()-t0
		Root0 = m.ObjBound
		del m
		del m_LP

		# Original formulation
		m = Model()
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.params.TimeLimit = tLimit
		m.params.LogFile = "LogFiles/MKAP/"+self.InstName+"_original_"+str(tLimit)+".log"
		m.update()
		t0 = time.time()
		m.optimize()
		t_IP0 = time.time()-t0
		if m.status == 2:
			ifSolve0 = 1
		else:
			ifSolve0 = 0
		nNodes0 = m.NodeCount
		Gap0 = m.MIPGap
		LB0 = m.ObjBound
		UB0 = m.objval
		del m

		t0_Dual = time.time()
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.params.SolutionLimit = 2
		m.update()
		m.optimize()
		UB = m.objval
		del m

		DualMaster = Model()
		DualMaster.modelSense = GRB.MAXIMIZE
		DualMaster.params.OutputFlag = 0
		DualMaster.params.threads = 4
		lam = DualMaster.addVars(self.n,lb=-float('inf'),ub=0)
		mu = DualMaster.addVars(self.m,lb=-float('inf'),ub=0)
		# theta <= V(lambda)
		theta = DualMaster.addVars(self.m,self.r,lb=-float('inf'))
		DualObj = DualMaster.addVar(lb=-float('inf'),ub=UB)
		DualMaster.addConstr(DualObj == quicksum(lam[j] for j in range(self.n))+quicksum(mu[i] for i in range(self.m))+quicksum(theta[i,k] for i in range(self.m) for k in range(self.r)))
		DualMaster.setObjective(DualObj)
		subIP = {}
		subx = {}
		suby = {}
		for i in range(self.m):
			for k in range(self.r):
				subIP[i,k] = Model()
				subIP[i,k].params.OutputFlag = 0
				subIP[i,k].params.threads = 4
				subx[i,k] = subIP[i,k].addVars(Sk[k],vtype=GRB.BINARY)
				suby[i,k] = subIP[i,k].addVar(vtype=GRB.BINARY)
				subIP[i,k].addConstr(quicksum(self.w[j]*subx[i,k][j] for j in Sk[k]) <= self.c[i]*suby[i,k])
		
		lam_soln = {}
		mu_soln = {}
		LB = -float('inf')
		iterN = 0
		StopCondt = False
		DualMethod = 'level'
		#DualMethod = 'cutpl'

		DWCuts = {(i,k):np.empty((0,len(Sk[k])+2)) for i in range(self.m) for k in range(self.r)}
		DualTime = 0
		subtime = 0
		LPtime = 0
		t0 = time.time()
		while StopCondt == False:
			iterN += 1
			print(iterN,LB,UB,time.time()-t0)
			DualMaster.update()
			DualMaster.optimize()
			UB = DualMaster.objval
			lam_old = lam_soln.copy()
			mu_old = mu_soln.copy()
			if DualMethod == 'cutpl' or (DualMethod == 'level' and iterN == 1):
				for j in range(self.n):
					lam_soln[j] = lam[j].x
				for i in range(self.m):
					mu_soln[i] = mu[i].x
			elif DualMethod == 'level':
				lt = UB-0.3*(UB-LB)
				levelCon = DualMaster.addConstr(DualObj >= lt)
				DualMaster.setObjective(-quicksum((lam[j]-lam_old[j])*(lam[j]-lam_old[j]) for j in range(self.n))-quicksum((mu[i]-mu_old[i])*(mu[i]-mu_old[i]) for i in range(self.m)))
				DualMaster.optimize()
				if DualMaster.status == 2:
					for j in range(self.n):
						lam_soln[j] = lam[j].x
					for i in range(self.m):
						mu_soln[i] = mu[i].x
				else:
					print('QP solver having numerical issues...')
					break
				DualMaster.remove(levelCon)
				DualMaster.setObjective(DualObj)
			if UB-LB < 1e-6*(min(abs(UB),abs(LB))+1):
				StopCondt = True
				print('Dual problem terminates!')

			# Solve "pricing" subproblem
			for i in range(self.m):
				for k in range(self.r):
					subIP[i,k].setObjective(-quicksum((self.p[j]+lam_soln[j])*subx[i,k][j] for j in Sk[k])-mu_soln[i]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					if subIP[i,k].status == 2:
						subx_soln = {j:subx[i,k][j].x for j in Sk[k]}
						suby_soln = suby[i,k].x
						DualMaster.addConstr(theta[i,k] <= -sum(self.p[j]*subx_soln[j] for j in Sk[k])-quicksum(subx_soln[j]*lam[j] for j in Sk[k])-suby_soln*mu[i])
						DWCuts[i,k] = np.append(DWCuts[i,k],np.array([[-self.p[j]-lam_soln[j] for j in Sk[k]]+[-mu_soln[i],-subIP[i,k].objval]]),axis=0)
					else:
						print('subIP error!')
			LB_new = sum(lam_soln[j] for j in range(self.n))+sum(mu_soln[i] for i in range(self.m))+sum(subIP[i,k].objval for i in range(self.m) for k in range(self.r))
			if LB_new > LB:
				LB = LB_new
		t_Dual = time.time()-t0_Dual
		subIPObj = {(i,k):subIP[i,k].objval for i in range(self.m) for k in range(self.r)}
		DualUB = UB
		DualLB = LB

		# Add as LB on the objective function
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.addConstr(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)) >= LB)
		#m.addConstrs((-quicksum((self.p[j]+lam_soln[j])*x[i,j] for j in Sk[k])-mu_soln[i]*y[i,k] >= subIPObj[i,k] for i in range(self.m) for k in range(self.r)))
		m.params.NodeLimit = 1
		m.update()
		t0 = time.time()
		m.optimize()
		Root1 = m.ObjBound
		RootTime1 = time.time()-t0
		del m

		m = Model()
		#m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.addConstr(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)) >= LB)
		m.update()
		m.params.TimeLimit = tLimit
		m.params.LogFile = "LogFiles/MKAP/"+self.InstName+"_objLB_"+str(tLimit)+".log"
		m.update()
		t0IP1 = time.time()
		m.optimize()
		t_IP1 = time.time()-t0IP1
		if m.status == 2:
			ifSolve1 = 1
		else:
			ifSolve1 = 0
		nNodes1 = m.NodeCount
		Gap1 = m.MIPGap
		LB1 = m.ObjBound
		UB1 = m.objval
		del m
		
		# Add block-by-block DWF cuts
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.addConstrs((-quicksum((self.p[j]+lam_soln[j])*x[i,j] for j in Sk[k])-mu_soln[i]*y[i,k] >= subIPObj[i,k] for i in range(self.m) for k in range(self.r)))
		m.params.NodeLimit = 1
		m.update()
		t0 = time.time()
		m.optimize()
		Root2 = m.ObjBound
		RootTime2 = time.time()-t0
		del m


		m = Model()
		#m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.addConstrs((-quicksum((self.p[j]+lam_soln[j])*x[i,j] for j in Sk[k])-mu_soln[i]*y[i,k] >= subIPObj[i,k] for i in range(self.m) for k in range(self.r)))
		m.params.TimeLimit = tLimit
		m.params.LogFile = "LogFiles/MKAP/"+self.InstName+"_DWcuts_"+str(tLimit)+".log"
		m.update()
		t0IP2 = time.time()
		m.optimize()
		t_IP2 = time.time()-t0IP2
		if m.status == 2:
			ifSolve2 = 1
		else:
			ifSolve2 = 0
		nNodes2 = m.NodeCount
		Gap2 = m.MIPGap
		LB2 = m.ObjBound
		UB2 = m.objval
		del m

		del DualMaster
		del subIP

		Output = [LB_LP,DualLB,DualUB,t_Dual,iterN,Root0,Root1,Root2,RootTime0,RootTime1,RootTime2,ifSolve0,ifSolve1,ifSolve2,t_IP0,t_IP1,t_IP2,nNodes0,nNodes1,nNodes2,\
			LB0,LB1,LB2,UB0,UB1,UB2,Gap0,Gap1,Gap2]
		
		OutputStr = ""
		for i in range(len(Output)):
			OutputStr = OutputStr+str(Output[i])+'\t'
		return OutputStr

	def CheckBounds(self):
		Sk = self.Sk
		# Calculate root bound
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.update()
		m_LP = m.relax()
		m_LP.update()
		m_LP.optimize()
		LB_LP = m_LP.ObjBound
		m.params.NodeLimit = 1
		m.update()
		t0 = time.time()
		m.optimize()
		RootTime0 = time.time()-t0
		Root0 = m.ObjBound
		del m
		del m_LP

		t0_Dual = time.time()
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.params.SolutionLimit = 2
		m.update()
		m.optimize()
		UB = m.objval
		del m

		DualMaster = Model()
		DualMaster.modelSense = GRB.MAXIMIZE
		DualMaster.params.OutputFlag = 0
		DualMaster.params.threads = 4
		lam = DualMaster.addVars(self.n,lb=-float('inf'),ub=0)
		mu = DualMaster.addVars(self.m,lb=-float('inf'),ub=0)
		# theta <= V(lambda)
		theta = DualMaster.addVars(self.m,self.r,lb=-float('inf'))
		DualObj = DualMaster.addVar(lb=-float('inf'),ub=UB)
		DualMaster.addConstr(DualObj == quicksum(lam[j] for j in range(self.n))+quicksum(mu[i] for i in range(self.m))+quicksum(theta[i,k] for i in range(self.m) for k in range(self.r)))
		DualMaster.setObjective(DualObj)
		subIP = {}
		subx = {}
		suby = {}
		for i in range(self.m):
			for k in range(self.r):
				subIP[i,k] = Model()
				subIP[i,k].params.OutputFlag = 0
				subIP[i,k].params.threads = 4
				subx[i,k] = subIP[i,k].addVars(Sk[k],vtype=GRB.BINARY)
				suby[i,k] = subIP[i,k].addVar(vtype=GRB.BINARY)
				subIP[i,k].addConstr(quicksum(self.w[j]*subx[i,k][j] for j in Sk[k]) <= self.c[i]*suby[i,k])
		
		lam_soln = {}
		mu_soln = {}
		LB = -float('inf')
		iterN = 0
		StopCondt = False
		DualMethod = 'level'
		#DualMethod = 'cutpl'

		DWCuts = {(i,k):np.empty((0,len(Sk[k])+2)) for i in range(self.m) for k in range(self.r)}
		DualTime = 0
		subtime = 0
		LPtime = 0
		t0 = time.time()
		while StopCondt == False:
			iterN += 1
			print(iterN,LB,UB,time.time()-t0)
			DualMaster.update()
			DualMaster.optimize()
			UB = DualMaster.objval
			lam_old = lam_soln.copy()
			mu_old = mu_soln.copy()
			if DualMethod == 'cutpl' or (DualMethod == 'level' and iterN == 1):
				for j in range(self.n):
					lam_soln[j] = lam[j].x
				for i in range(self.m):
					mu_soln[i] = mu[i].x
			elif DualMethod == 'level':
				lt = UB-0.3*(UB-LB)
				levelCon = DualMaster.addConstr(DualObj >= lt)
				DualMaster.setObjective(-quicksum((lam[j]-lam_old[j])*(lam[j]-lam_old[j]) for j in range(self.n))-quicksum((mu[i]-mu_old[i])*(mu[i]-mu_old[i]) for i in range(self.m)))
				DualMaster.optimize()
				if DualMaster.status == 2:
					for j in range(self.n):
						lam_soln[j] = lam[j].x
					for i in range(self.m):
						mu_soln[i] = mu[i].x
				else:
					print('QP solver having numerical issues...')
					break
				DualMaster.remove(levelCon)
				DualMaster.setObjective(DualObj)
			if UB-LB < 1e-6*(min(abs(UB),abs(LB))+1):
				StopCondt = True
				print('Dual problem terminates!')

			# Solve "pricing" subproblem
			for i in range(self.m):
				for k in range(self.r):
					subIP[i,k].setObjective(-quicksum((self.p[j]+lam_soln[j])*subx[i,k][j] for j in Sk[k])-mu_soln[i]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					if subIP[i,k].status == 2:
						subx_soln = {j:subx[i,k][j].x for j in Sk[k]}
						suby_soln = suby[i,k].x
						DualMaster.addConstr(theta[i,k] <= -sum(self.p[j]*subx_soln[j] for j in Sk[k])-quicksum(subx_soln[j]*lam[j] for j in Sk[k])-suby_soln*mu[i])
						DWCuts[i,k] = np.append(DWCuts[i,k],np.array([[-self.p[j]-lam_soln[j] for j in Sk[k]]+[-mu_soln[i],-subIP[i,k].objval]]),axis=0)
					else:
						print('subIP error!')
			LB_new = sum(lam_soln[j] for j in range(self.n))+sum(mu_soln[i] for i in range(self.m))+sum(subIP[i,k].objval for i in range(self.m) for k in range(self.r))
			if LB_new > LB:
				LB = LB_new
		t_Dual = time.time()-t0_Dual
		subIPObj = {(i,k):subIP[i,k].objval for i in range(self.m) for k in range(self.r)}
		DualUB = UB
		DualLB = LB

		return LB_LP,Root0,DualLB,DualUB,RootTime0,t_Dual

	def CompareCTilt(self,tLimit=10*60):
		#Sk = {k:[k*int(self.n/self.r)+p for p in range(int(self.n/self.r))] for k in range(self.r)}
		Sk = self.Sk
		t0_Dual = time.time()
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.params.SolutionLimit = 2
		m.update()
		m.optimize()
		UB = m.objval
		del m

		DualMaster = Model()
		DualMaster.modelSense = GRB.MAXIMIZE
		DualMaster.params.OutputFlag = 0
		DualMaster.params.threads = 4
		lam = DualMaster.addVars(self.n,lb=-float('inf'),ub=0)
		mu = DualMaster.addVars(self.m,lb=-float('inf'),ub=0)
		# theta <= V(lambda)
		theta = DualMaster.addVars(self.m,self.r,lb=-float('inf'))
		DualObj = DualMaster.addVar(lb=-float('inf'),ub=UB)
		DualMaster.addConstr(DualObj == quicksum(lam[j] for j in range(self.n))+quicksum(mu[i] for i in range(self.m))+quicksum(theta[i,k] for i in range(self.m) for k in range(self.r)))
		DualMaster.setObjective(DualObj)
		subIP = {}
		subx = {}
		suby = {}
		for i in range(self.m):
			for k in range(self.r):
				subIP[i,k] = Model()
				subIP[i,k].params.OutputFlag = 0
				subIP[i,k].params.threads = 4
				subx[i,k] = subIP[i,k].addVars(Sk[k],vtype=GRB.BINARY)
				suby[i,k] = subIP[i,k].addVar(vtype=GRB.BINARY)
				subIP[i,k].addConstr(quicksum(self.w[j]*subx[i,k][j] for j in Sk[k]) <= self.c[i]*suby[i,k])
		
		lam_soln = {}
		mu_soln = {}
		LB = -float('inf')
		iterN = 0
		StopCondt = False
		DualMethod = 'level'
		#DualMethod = 'cutpl'
		nSubIPsolved = 0

		DWCuts = {(i,k):np.empty((0,len(Sk[k])+2)) for i in range(self.m) for k in range(self.r)}
		SubIPSolns = {(i,k):np.empty((0,len(Sk[k])+2)) for i in range(self.m) for k in range(self.r)}
		DualTime = 0
		subtime = 0
		LPtime = 0
		t0 = time.time()
		while StopCondt == False:
			iterN += 1
			print(iterN,LB,UB,time.time()-t0)
			DualMaster.update()
			DualMaster.optimize()
			UB = DualMaster.objval
			lam_old = lam_soln.copy()
			mu_old = mu_soln.copy()
			if DualMethod == 'cutpl' or (DualMethod == 'level' and iterN == 1):
				for j in range(self.n):
					lam_soln[j] = lam[j].x
				for i in range(self.m):
					mu_soln[i] = mu[i].x
			elif DualMethod == 'level':
				lt = UB-0.3*(UB-LB)
				levelCon = DualMaster.addConstr(DualObj >= lt)
				DualMaster.setObjective(-quicksum((lam[j]-lam_old[j])*(lam[j]-lam_old[j]) for j in range(self.n))-quicksum((mu[i]-mu_old[i])*(mu[i]-mu_old[i]) for i in range(self.m)))
				DualMaster.optimize()
				if DualMaster.status == 2:
					for j in range(self.n):
						lam_soln[j] = lam[j].x
					for i in range(self.m):
						mu_soln[i] = mu[i].x
				else:
					print('QP solver having numerical issues...')
					break
				DualMaster.remove(levelCon)
				DualMaster.setObjective(DualObj)
			if UB-LB < 1e-6*(min(abs(UB),abs(LB))+1):
				StopCondt = True
				print('Dual problem terminates!')

			# Solve "pricing" subproblem
			for i in range(self.m):
				for k in range(self.r):
					subIP[i,k].setObjective(-quicksum((self.p[j]+lam_soln[j])*subx[i,k][j] for j in Sk[k])-mu_soln[i]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					nSubIPsolved += 1
					if subIP[i,k].status == 2:
						subx_soln = {j:subx[i,k][j].x for j in Sk[k]}
						suby_soln = suby[i,k].x
						DualMaster.addConstr(theta[i,k] <= -sum(self.p[j]*subx_soln[j] for j in Sk[k])-quicksum(subx_soln[j]*lam[j] for j in Sk[k])-suby_soln*mu[i])
						DWCuts[i,k] = np.append(DWCuts[i,k],np.array([[-self.p[j]-lam_soln[j] for j in Sk[k]]+[-mu_soln[i],-subIP[i,k].objval]]),axis=0)
						soln = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x,1.0]
						if soln not in SubIPSolns[i,k].tolist():
							SubIPSolns[i,k]=np.append(SubIPSolns[i,k],[soln],axis=0)
					else:
						print('subIP error!')
			LB_new = sum(lam_soln[j] for j in range(self.n))+sum(mu_soln[i] for i in range(self.m))+sum(subIP[i,k].objval for i in range(self.m) for k in range(self.r))
			if LB_new > LB:
				LB = LB_new
		t_Dual = time.time()-t0_Dual
		subIPObj = {(i,k):subIP[i,k].objval for i in range(self.m) for k in range(self.r)}
		DualUB = UB
		DualLB = LB

		#Depth = min(6,int(self.n/self.r)-1) # Test up to depth 6 tilting
		DepthUB = 6
		TightAtCurrentNode = {(i,k,0,0):([[subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]],None) for i in range(self.m) for k in range(self.r)}
		

		CTiltCut = {}
		nTilt = 0
		nCSubIPs = 0
		t0_CTilt = time.time()
		for i in range(self.m):
			for k in range(self.r):
				cut = {}
				for j in Sk[k]:
					cut[j] = -self.p[j]-lam_soln[j]
				cut[self.n] = -mu_soln[i]
				cutRHS = subIP[i,k].objval
				CoordStatus = {}
				for j in Sk[k]:
					if subx[i,k][j].x > 0.5:
						CoordStatus[j] = 1
					else:
						CoordStatus[j] = 0
				if suby[i,k].x > 0.5:
					CoordStatus[self.n] = 1
				else:
					CoordStatus[self.n] = 0

				# CoordStatus[j] = 0(/1) means all known feasible points have its j-th coordinate == 0(/1)
				# CoordStatus[j] = 2 means there are known feasible points having its j-th coordinate equal to 0 and equal to 1, in which case one cannot tilt

				for j in Sk[k]:
					if CoordStatus[j] == 1:
						subx[i,k][j].ub = 0.0
						subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
						subIP[i,k].update()
						subIP[i,k].optimize()
						nCSubIPs += 1
						TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
						delta = subIP[i,k].objval - cutRHS
						if delta > 1e-6:
							cutRHS += delta
							cut[j] += delta
							nTilt += 1
						for jj in Sk[k]:
							if CoordStatus[jj] == 0 and subx[i,k][jj].x > 0.5:
								CoordStatus[jj] = 2
							elif CoordStatus[jj] == 1 and subx[i,k][jj].x < 0.5:
								CoordStatus[jj] = 2
						if CoordStatus[self.n] == 0 and suby[i,k].x > 0.5:
							CoordStatus[self.n] = 2
						elif CoordStatus[self.n] == 1 and suby[i,k].x < 0.5:
							CoordStatus[self.n] = 2
						subx[i,k][j].ub = 1.0
						subIP[i,k].update()
					elif CoordStatus[j] == 0 and self.w[j] <= self.c[i]:
						subx[i,k][j].lb = 1.0
						subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
						subIP[i,k].update()
						subIP[i,k].optimize()
						nCSubIPs += 1
						TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
						delta = subIP[i,k].objval - cutRHS
						if delta > 1e-6:
							cut[j] -= delta
							nTilt += 1
						for jj in Sk[k]:
							if CoordStatus[jj] == 0 and subx[i,k][jj].x > 0.5:
								CoordStatus[jj] = 2
							elif CoordStatus[jj] == 1 and subx[i,k][jj].x < 0.5:
								CoordStatus[jj] = 2
						if CoordStatus[self.n] == 0 and suby[i,k].x > 0.5:
							CoordStatus[self.n] = 2
						elif CoordStatus[self.n] == 1 and suby[i,k].x < 0.5:
							CoordStatus[self.n] = 2
						subx[i,k][j].lb = 0.0
						subIP[i,k].update()

				if CoordStatus[self.n] == 1:
					suby[i,k].ub = 0.0
					subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					nCSubIPs += 1
					TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
					delta = subIP[i,k].objval - cutRHS
					if delta > 1e-6:
						cutRHS += delta
						cut[self.n] += delta
						nTilt += 1
					suby[i,k].ub = 1.0
					subIP[i,k].update()
				elif CoordStatus[self.n] == 0:
					suby[i,k].lb = 1.0
					subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					nCSubIPs += 1
					TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
					delta = subIP[i,k].objval - cutRHS
					if delta > 1e-6:
						cut[self.n] -= delta
						nTilt += 1
					suby[i,k].lb = 0.0
					subIP[i,k].update()

				CTiltCut[i,k] = [cut[j] for j in Sk[k]]+[cut[self.n],-cutRHS]
		t_CTilt = time.time()-t0_CTilt

		t0 = time.time()
		TiltTime =[]
		Depth = {(i,k): max(min(DepthUB,len([j for j in Sk[k] if self.w[j] <= self.c[i]])+1-len(TightAtCurrentNode[i,k,0,0][0])),0) for i in range(self.m) for k in range(self.r)}		
		TiltedCuts = {(i,k,d):[] for i in range(self.m) for k in range(self.r) for d in range(Depth[i,k]+1)}
		for i in range(self.m):
			for k in range(self.r):
				TiltedCuts[i,k,0].append(CTiltCut[i,k].copy())
		

		# Tilt the cuts
		for d in range(DepthUB):
			if d >= 1:
				TiltTime.append(time.time()-t0)
			for i in range(self.m):
				for k in range(self.r):
					#if i == 0 and k == 1:
						#print(TightAtCurrentNode[i,k,0,0][0])
						#for poi in TightAtCurrentNode[i,k,0,0][0]:
						#	print(sum(poi[j]*CTiltCut[i,k][j] for j in range(len(poi)))+CTiltCut[i,k][-1])
						#print(CTiltCut[i,k])
					if d < Depth[i,k]:
						#print(i,k,d)
						for cutInd in range(len(TiltedCuts[i,k,d])):
							cut = TiltedCuts[i,k,d][cutInd]
							tightInd = (i,k,d,cutInd)
							tight = TightAtCurrentNode[tightInd][0]
							while TightAtCurrentNode[tightInd][1] != None:
								tightInd = TightAtCurrentNode[tightInd][1]
								tight = tight+TightAtCurrentNode[tightInd][0]
							SearchLP = Model()
							SearchLP.params.OutputFlag = 0
							SearchLP.params.threads = 4
							v = SearchLP.addVars(int(self.n/self.r)+1,lb=-1,ub=1)
							w = SearchLP.addVar()
							for kk in range(int(self.n/self.r)):
								if self.w[Sk[k][kk]] > self.c[i]:
									SearchLP.addConstr(v[kk] == 0)
							for poi in tight:
								SearchLP.addConstr(quicksum(poi[ii]*v[ii] for ii in range(int(self.n/self.r)+1)) == w)
							NegVio = SubIPSolns[i,k].dot(cut)
							FoundCand = False
							for kk in range(len(NegVio)):
								if NegVio[-kk] > 1e-6:
									cand = SubIPSolns[i,k][-kk].copy()
									FoundCand = True
									break

							if FoundCand == False:
								subIP[i,k].setObjective(-quicksum(cut[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))-cut[int(self.n/self.r)]*suby[i,k])
								subIP[i,k].update()
								subIP[i,k].optimize()
								nSubIPsolved += 1
								cand = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x,1]

							# Search for tilting direction
							CandCon = SearchLP.addConstr(quicksum(cand[ii]*v[ii] for ii in range(int(self.n/self.r)+1)) == w)
							FoundDir = False
							aim = 0
							sign = 1
							while FoundDir == False:
								# Optimize over direction e_1, e_2,..., e_n, -e_1, -e_2,..., -e_N
								SearchLP.setObjective(sign*v[aim])
								SearchLP.update()
								SearchLP.optimize()
								#print(SearchLP.NumConstrs)
								Dir = [v[i].x for i in range(int(self.n/self.r)+1)]+[w.x]
								if sum(abs(Dir[i]) for i in range(int(self.n/self.r)+1)) <= 1e-8:
									if aim < int(self.n/self.r):
										aim += 1
									else:
										aim = 0
										sign = -1
								else:
									FoundDir = True

							vk1 = [v[ii].x for ii in range(int(self.n/self.r)+1)]
							wk1 = w.x
							vk2 = [-v[ii].x for ii in range(int(self.n/self.r)+1)]
							wk2 = -w.x
							FoundValid = False
							lamk1 = float('inf')
							lamk2 = float('inf')
							new_soln1 = [cand[ii] for ii in range(int(self.n/self.r)+1)]
							new_soln2 = [cand[ii] for ii in range(int(self.n/self.r)+1)]

							# Use known integer solutions to initialize lamk1 and lamk2
							abVio = SubIPSolns[i,k].dot(cut)
							vwVio = SubIPSolns[i,k].dot([v[ii].x for ii in range(int(self.n/self.r)+1)]+[-w.x])
							for kk in range(len(SubIPSolns[i,k])):
								if vwVio[kk] < -1e-8 and lamk1>-abVio[kk]/vwVio[kk]:
									lamk1 = -abVio[kk]/vwVio[kk]
									new_soln1 = [SubIPSolns[i,k][kk][ii] for ii in range(int(self.n/self.r)+1)]
								elif vwVio[kk] > 1e-8 and lamk2>abVio[kk]/vwVio[kk]:
									lamk2 = abVio[kk]/vwVio[kk]
									new_soln2 = [SubIPSolns[i,k][kk][ii] for ii in range(int(self.n/self.r)+1)]
							if lamk1 < float('inf') and lamk1 >= 1e-8:
								vk1 = [cut[ii]+lamk1*v[ii].x for ii in range(int(self.n/self.r)+1)]
								wk1 = -cut[int(self.n/self.r)+1]+lamk1*w.x
							elif lamk1 < 1e-8:
								vk1 = [cut[ii] for ii in range(int(self.n/self.r)+1)]
								wk1 = -cut[-1]
							if lamk2 < float('inf') and lamk2 >= 1e-8:
								vk2 = [cut[ii]-lamk2*v[ii].x for ii in range(int(self.n/self.r)+1)]
								wk2 = -cut[int(self.n/self.r)+1]-lamk2*w.x
							elif lamk2 < 1e-8:
								vk2 = [cut[ii] for ii in range(int(self.n/self.r)+1)]
								wk2 = -cut[-1]

							lamk1old = lamk1
							lamk2old = lamk2

							if lamk1 >= 1e-8:
								# Diretion v^Tx >= w
								while FoundValid == False:
									#print('lamk1',lamk1)
									if lamk1 < 1e-8 or (lamk1 > lamk1old and lamk1old < 1e-5):
										FoundValid = True
										vk1 = [cut[ii] for ii in range(int(self.n/self.r)+1)]
										wk1 = -cut[-1]
										new_soln1 = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]
									else:
										subIP[i,k].setObjective(quicksum(vk1[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+vk1[int(self.n/self.r)]*suby[i,k])
										subIP[i,k].update()
										subIP[i,k].optimize()
										nSubIPsolved += 1
										soln = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x,1.0]
										if soln not in SubIPSolns[i,k].tolist():
											SubIPSolns[i,k]=np.append(SubIPSolns[i,k],[soln],axis=0)
										#SubIPSolns[j]=np.append(SubIPSolns[j],[[subx[j][i].x for i in range(self.N)]+[suby[j].x,1.0]],axis=0)
										if subIP[i,k].objval < wk1-1e-6:
											# vk= a+lamk1*v, wk = b+lamk1*w, lamk1 = (a^Tx^*-b)/(w-v^Tx^*)
											if abs(sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x) < 1e-5 and subIP[i,k].objval >= wk1-1e-3:
												#print(i,k,d,cutInd,len(TiltedCuts[i,k,d+1]))
												wk1 = subIP[i,k].objval
												FoundValid = True
												break
											lamk1old = lamk1
											lamk1 = (sum(cut[ii]*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*suby[i,k].x+cut[int(self.n/self.r)+1])/(w.x-sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))-v[int(self.n/self.r)].x*suby[i,k].x)
											vk1 = [cut[ii]+lamk1*v[ii].x for ii in range(int(self.n/self.r)+1)]
											wk1 = -cut[int(self.n/self.r)+1]+lamk1*w.x
											new_soln1 = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]
										else:
											wk1 = subIP[i,k].objval
											FoundValid = True
							
							#if i == 0 and k == 2 and d == 4 and cutInd == 13:
							#	subIP[i,k].setObjective(quicksum(cut[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*suby[i,k])
							#	subIP[i,k].update()
							#	subIP[i,k].optimize()
							#	print(subIP[i,k].objval,cut[-1])

							if lamk2 >= 1e-8:
								# Diretion -v^Tx >= -w
								FoundValid = False
								while FoundValid == False:
									if lamk2 < 1e-8 or (lamk2 > lamk2old and lamk2old < 1e-5):
										FoundValid = True
										vk2 = [cut[ii] for ii in range(int(self.n/self.r)+1)]
										wk2 = -cut[-1]
										new_soln2 = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]
									else:
										subIP[i,k].setObjective(quicksum(vk2[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+vk2[int(self.n/self.r)]*suby[i,k])
										subIP[i,k].update()
										subIP[i,k].optimize()
										#if i == 0 and k == 2 and d == 4 and cutInd == 13:
										#	print(sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x,\
										#		subIP[i,k].objval,wk2,lamk2,\
										#		sum(cut[ii]*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*suby[i,k].x+cut[int(self.n/self.r)+1],\
										#		sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x)
										nSubIPsolved += 1
										soln = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x,1.0]
										if soln not in SubIPSolns[i,k].tolist():
											SubIPSolns[i,k]=np.append(SubIPSolns[i,k],[soln],axis=0)
										if subIP[i,k].objval < wk2-1e-6:
											# vk= a-lamk2*v, wk = b-lamk2*w, lamk2 = (a^Tx^*-b)/(v^Tx^*-w)
											if abs(sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x) < 1e-5 and subIP[i,k].objval >= wk2-1e-3:
												#print(i,k,d,cutInd,len(TiltedCuts[i,k,d+1]))
												wk2 = subIP[i,k].objval
												FoundValid = True
												break
											lamk2old = lamk2
											lamk2 = (sum(cut[ii]*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*suby[i,k].x+cut[int(self.n/self.r)+1])/(sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x)
											vk2 = [cut[ii]-lamk2*v[ii].x for ii in range(int(self.n/self.r)+1)]
											wk2 = -cut[int(self.n/self.r)+1]-lamk2*w.x
											new_soln2 = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]
										else:
											wk2 = subIP[i,k].objval
											FoundValid = True

							TiltedCuts[i,k,d+1].append(vk1+[-wk1])
							TightAtCurrentNode[i,k,d+1,len(TiltedCuts[i,k,d+1])-1] = ([new_soln1],(i,k,d,cutInd))
							#subIP[i,k].setObjective(quicksum(vk1[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+vk1[int(self.n/self.r)]*suby[i,k])
							#subIP[i,k].update()
							#subIP[i,k].optimize()
							#if subIP[i,k].objval < wk1-1e-8:
							#	print(d,i,k,lamk1,len(TiltedCuts[i,k,d+1])-1,subIP[i,k].objval,wk1,'*')

							if lamk1 >= 1e-8 or lamk2 >= 1e-8:
								TiltedCuts[i,k,d+1].append(vk2+[-wk2])
								TightAtCurrentNode[i,k,d+1,len(TiltedCuts[i,k,d+1])-1] = ([new_soln2],(i,k,d,cutInd))
								#subIP[i,k].setObjective(quicksum(vk2[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+vk2[int(self.n/self.r)]*suby[i,k])
								#subIP[i,k].update()
								#subIP[i,k].optimize()
								#if subIP[i,k].objval < wk2-1e-8:
								#	print(d,i,k,lamk2,len(TiltedCuts[i,k,d+1])-1,subIP[i,k].objval,wk2,'**')
							
		TiltTime.append(time.time()-t0)
		
		
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.addConstrs((quicksum(CTiltCut[i,k][ii]*x[i,Sk[k][ii]] for ii in range(int(self.n/self.r)))+CTiltCut[i,k][int(self.n/self.r)]*y[i,k] >= -CTiltCut[i,k][int(self.n/self.r)+1]-1e-5 for i in range(self.m) for k in range(self.r)))
		m.params.NodeLimit = 1
		m.update()
		t0 = time.time()
		m.optimize()
		RootCT = m.ObjBound
		RootTimeCT = time.time()-t0
		del m


		m = Model()
		#m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.addConstrs((quicksum(CTiltCut[i,k][ii]*x[i,Sk[k][ii]] for ii in range(int(self.n/self.r)))+CTiltCut[i,k][int(self.n/self.r)]*y[i,k] >= -CTiltCut[i,k][int(self.n/self.r)+1]-1e-5 for i in range(self.m) for k in range(self.r)))
		m.params.TimeLimit = tLimit
		m.params.LogFile = "LogFiles/MKAP/"+self.InstName+"_DWcutsCTp_"+str(tLimit)+".log"
		m.update()
		t0 = time.time()
		m.optimize()
		t_IPCT = time.time()-t0
		if m.status == 2:
			ifSolveCT = 1
		else:
			ifSolveCT = 0
		nNodesCT = m.NodeCount
		GapCT = m.MIPGap
		LBCT = m.ObjBound
		UBCT = m.objval
		del m

		RootCTd = []
		RootTimeCTd = []
		t_IPCTd = []
		ifSolveCTd = []
		nNodesCTd = []
		GapCTd = []
		LBCTd = []
		UBCTd = []

		
		for d in range(1,DepthUB+1):
			m = Model()
			m.params.OutputFlag = 0
			m.params.threads = 4
			x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
			y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
			m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
			m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
			m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
			m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
			NewCuts = []
			nCuts = 0
			for i in range(self.m):
				for k in range(self.r):
					for cut in TiltedCuts[i,k,min(d,Depth[i,k])]:
						NewCuts.append(m.addConstr(quicksum(cut[ii]*x[i,Sk[k][ii]] for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*y[i,k] >= -cut[int(self.n/self.r)+1]-1e-5))
						#NewCuts[nCuts].setAttr('lazy',-1)
						nCuts += 1
			m.params.NodeLimit = 1
			m.update()
			t0 = time.time()
			m.optimize()
			RootCTd.append(m.ObjBound)
			RootTimeCTd.append(time.time()-t0)
			del m

			m = Model()
			#m.params.OutputFlag = 0
			m.params.threads = 4
			x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
			y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
			m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
			m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
			m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
			m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
			NewCuts = []
			nCuts = 0
			for i in range(self.m):
				for k in range(self.r):
					for cut in TiltedCuts[i,k,min(d,Depth[i,k])]:
						NewCuts.append(m.addConstr(quicksum(cut[ii]*x[i,Sk[k][ii]] for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*y[i,k] >= -cut[int(self.n/self.r)+1]-1e-5))
						#NewCuts[nCuts].setAttr('lazy',-1)
						nCuts += 1
			m.params.TimeLimit = tLimit
			m.params.LogFile = "LogFiles/MKAP/"+self.InstName+"_DWcutsCTp"+str(d)+"_"+str(tLimit)+".log"
			m.update()
			t0 = time.time()
			m.optimize()
			t_IPCTd.append(time.time()-t0)
			if m.status == 2:
				ifSolveCTd.append(1)
			else:
				ifSolveCTd.append(0)
			nNodesCTd.append(m.NodeCount)
			GapCTd.append(m.MIPGap)
			LBCTd.append(m.ObjBound)
			UBCTd.append(m.objval)
			del m

		del DualMaster
		del subIP

		Output = [t_CTilt,nTilt,nCSubIPs,RootCT,RootTimeCT,ifSolveCT,t_IPCT,nNodesCT,LBCT,UBCT,GapCT]+TiltTime+RootCTd+ifSolveCTd+t_IPCTd+nNodesCTd+LBCTd+UBCTd+GapCTd
		
		OutputStr = ""
		for i in range(len(Output)):
			OutputStr = OutputStr+str(Output[i])+'\t'
		return OutputStr
		
	def ComputeDim(self):
		dim_OPTface = []
		Sk = {k:[k*int(self.n/self.r)+p for p in range(int(self.n/self.r))] for k in range(self.r)}
		# Calculate root bound
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,lb=0.0,ub=1.0)
		y = m.addVars(self.m,self.r,lb=0.0,ub=1.0)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.update()
		m.optimize()
		constrs = m.getConstrs()
		variables = m.getVars()

		pis = [con.pi for con in constrs]
		rdcosts = [var.RC for var in variables]

		dim_OPTface.append(self.m*self.n+self.m*self.r-len([pi for pi in pis if abs(pi) > 1e-8])-len([rdcost for rdcost in rdcosts if abs(rdcost) > 1e-8]))


		Sk = self.Sk
		t0_Dual = time.time()
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.params.SolutionLimit = 2
		m.update()
		m.optimize()
		UB = m.objval
		del m
		
		DualMaster = Model()
		DualMaster.modelSense = GRB.MAXIMIZE
		DualMaster.params.OutputFlag = 0
		DualMaster.params.threads = 4
		lam = DualMaster.addVars(self.n,lb=-float('inf'),ub=0)
		mu = DualMaster.addVars(self.m,lb=-float('inf'),ub=0)
		# theta <= V(lambda)
		theta = DualMaster.addVars(self.m,self.r,lb=-float('inf'))
		DualObj = DualMaster.addVar(lb=-float('inf'),ub=UB)
		DualMaster.addConstr(DualObj == quicksum(lam[j] for j in range(self.n))+quicksum(mu[i] for i in range(self.m))+quicksum(theta[i,k] for i in range(self.m) for k in range(self.r)))
		DualMaster.setObjective(DualObj)
		subIP = {}
		subx = {}
		suby = {}
		for i in range(self.m):
			for k in range(self.r):
				subIP[i,k] = Model()
				subIP[i,k].params.OutputFlag = 0
				subIP[i,k].params.threads = 4
				subx[i,k] = subIP[i,k].addVars(Sk[k],vtype=GRB.BINARY)
				suby[i,k] = subIP[i,k].addVar(vtype=GRB.BINARY)
				subIP[i,k].addConstr(quicksum(self.w[j]*subx[i,k][j] for j in Sk[k]) <= self.c[i]*suby[i,k])
		
		lam_soln = {}
		mu_soln = {}
		LB = -float('inf')
		iterN = 0
		StopCondt = False
		DualMethod = 'level'
		#DualMethod = 'cutpl'
		nSubIPsolved = 0

		DWCuts = {(i,k):np.empty((0,len(Sk[k])+2)) for i in range(self.m) for k in range(self.r)}
		SubIPSolns = {(i,k):np.empty((0,len(Sk[k])+2)) for i in range(self.m) for k in range(self.r)}
		DualTime = 0
		subtime = 0
		LPtime = 0
		t0 = time.time()
		while StopCondt == False:
			iterN += 1
			print(iterN,LB,UB,time.time()-t0)
			DualMaster.update()
			DualMaster.optimize()
			UB = DualMaster.objval
			lam_old = lam_soln.copy()
			mu_old = mu_soln.copy()
			if DualMethod == 'cutpl' or (DualMethod == 'level' and iterN == 1):
				for j in range(self.n):
					lam_soln[j] = lam[j].x
				for i in range(self.m):
					mu_soln[i] = mu[i].x
			elif DualMethod == 'level':
				lt = UB-0.3*(UB-LB)
				levelCon = DualMaster.addConstr(DualObj >= lt)
				DualMaster.setObjective(-quicksum((lam[j]-lam_old[j])*(lam[j]-lam_old[j]) for j in range(self.n))-quicksum((mu[i]-mu_old[i])*(mu[i]-mu_old[i]) for i in range(self.m)))
				DualMaster.optimize()
				if DualMaster.status == 2:
					for j in range(self.n):
						lam_soln[j] = lam[j].x
					for i in range(self.m):
						mu_soln[i] = mu[i].x
				else:
					print('QP solver having numerical issues...')
					break
				DualMaster.remove(levelCon)
				DualMaster.setObjective(DualObj)
			if UB-LB < 1e-6*(min(abs(UB),abs(LB))+1):
				StopCondt = True
				print('Dual problem terminates!')

			# Solve "pricing" subproblem
			for i in range(self.m):
				for k in range(self.r):
					subIP[i,k].setObjective(-quicksum((self.p[j]+lam_soln[j])*subx[i,k][j] for j in Sk[k])-mu_soln[i]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					nSubIPsolved += 1
					if subIP[i,k].status == 2:
						subx_soln = {j:subx[i,k][j].x for j in Sk[k]}
						suby_soln = suby[i,k].x
						DualMaster.addConstr(theta[i,k] <= -sum(self.p[j]*subx_soln[j] for j in Sk[k])-quicksum(subx_soln[j]*lam[j] for j in Sk[k])-suby_soln*mu[i])
						DWCuts[i,k] = np.append(DWCuts[i,k],np.array([[-self.p[j]-lam_soln[j] for j in Sk[k]]+[-mu_soln[i],-subIP[i,k].objval]]),axis=0)
						soln = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x,1.0]
						if soln not in SubIPSolns[i,k].tolist():
							SubIPSolns[i,k]=np.append(SubIPSolns[i,k],[soln],axis=0)
					else:
						print('subIP error!')
			LB_new = sum(lam_soln[j] for j in range(self.n))+sum(mu_soln[i] for i in range(self.m))+sum(subIP[i,k].objval for i in range(self.m) for k in range(self.r))
			if LB_new > LB:
				LB = LB_new
		t_Dual = time.time()-t0_Dual
		subIPObj = {(i,k):subIP[i,k].objval for i in range(self.m) for k in range(self.r)}
		DualUB = UB
		DualLB = LB

		#Depth = min(6,int(self.n/self.r)-1) # Test up to depth 6 tilting
		DepthUB = 6
		TightAtCurrentNode = {(i,k,0,0):([[subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]],None) for i in range(self.m) for k in range(self.r)}
		

		CTiltCut = {}
		nTilt = 0
		nCSubIPs = 0
		t0_CTilt = time.time()
		for i in range(self.m):
			for k in range(self.r):
				cut = {}
				for j in Sk[k]:
					cut[j] = -self.p[j]-lam_soln[j]
				cut[self.n] = -mu_soln[i]
				cutRHS = subIP[i,k].objval
				CoordStatus = {}
				for j in Sk[k]:
					if subx[i,k][j].x > 0.5:
						CoordStatus[j] = 1
					else:
						CoordStatus[j] = 0
				if suby[i,k].x > 0.5:
					CoordStatus[self.n] = 1
				else:
					CoordStatus[self.n] = 0

				# CoordStatus[j] = 0(/1) means all known feasible points have its j-th coordinate == 0(/1)
				# CoordStatus[j] = 2 means there are known feasible points having its j-th coordinate equal to 0 and equal to 1, in which case one cannot tilt

				for j in Sk[k]:
					if CoordStatus[j] == 1:
						subx[i,k][j].ub = 0.0
						subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
						subIP[i,k].update()
						subIP[i,k].optimize()
						nCSubIPs += 1
						TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
						delta = subIP[i,k].objval - cutRHS
						if delta > 1e-6:
							cutRHS += delta
							cut[j] += delta
							nTilt += 1
						for jj in Sk[k]:
							if CoordStatus[jj] == 0 and subx[i,k][jj].x > 0.5:
								CoordStatus[jj] = 2
							elif CoordStatus[jj] == 1 and subx[i,k][jj].x < 0.5:
								CoordStatus[jj] = 2
						if CoordStatus[self.n] == 0 and suby[i,k].x > 0.5:
							CoordStatus[self.n] = 2
						elif CoordStatus[self.n] == 1 and suby[i,k].x < 0.5:
							CoordStatus[self.n] = 2
						subx[i,k][j].ub = 1.0
						subIP[i,k].update()
					elif CoordStatus[j] == 0 and self.w[j] <= self.c[i]:
						subx[i,k][j].lb = 1.0
						subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
						subIP[i,k].update()
						subIP[i,k].optimize()
						nCSubIPs += 1
						TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
						delta = subIP[i,k].objval - cutRHS
						if delta > 1e-6:
							cut[j] -= delta
							nTilt += 1
						for jj in Sk[k]:
							if CoordStatus[jj] == 0 and subx[i,k][jj].x > 0.5:
								CoordStatus[jj] = 2
							elif CoordStatus[jj] == 1 and subx[i,k][jj].x < 0.5:
								CoordStatus[jj] = 2
						if CoordStatus[self.n] == 0 and suby[i,k].x > 0.5:
							CoordStatus[self.n] = 2
						elif CoordStatus[self.n] == 1 and suby[i,k].x < 0.5:
							CoordStatus[self.n] = 2
						subx[i,k][j].lb = 0.0
						subIP[i,k].update()

				if CoordStatus[self.n] == 1:
					suby[i,k].ub = 0.0
					subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					nCSubIPs += 1
					TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
					delta = subIP[i,k].objval - cutRHS
					if delta > 1e-6:
						cutRHS += delta
						cut[self.n] += delta
						nTilt += 1
					suby[i,k].ub = 1.0
					subIP[i,k].update()
				elif CoordStatus[self.n] == 0:
					suby[i,k].lb = 1.0
					subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					nCSubIPs += 1
					TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
					delta = subIP[i,k].objval - cutRHS
					if delta > 1e-6:
						cut[self.n] -= delta
						nTilt += 1
					suby[i,k].lb = 0.0
					subIP[i,k].update()

				CTiltCut[i,k] = [cut[j] for j in Sk[k]]+[cut[self.n],-cutRHS]
		t_CTilt = time.time()-t0_CTilt

		t0 = time.time()
		TiltTime =[]
		Depth = {(i,k): max(min(DepthUB,len([j for j in Sk[k] if self.w[j] <= self.c[i]])+1-len(TightAtCurrentNode[i,k,0,0][0])),0) for i in range(self.m) for k in range(self.r)}		
		TiltedCuts = {(i,k,d):[] for i in range(self.m) for k in range(self.r) for d in range(Depth[i,k]+1)}
		for i in range(self.m):
			for k in range(self.r):
				TiltedCuts[i,k,0].append(CTiltCut[i,k].copy())
		

		# Tilt the cuts
		for d in range(DepthUB):
			if d >= 1:
				TiltTime.append(time.time()-t0)
			for i in range(self.m):
				for k in range(self.r):
					#if i == 0 and k == 1:
						#print(TightAtCurrentNode[i,k,0,0][0])
						#for poi in TightAtCurrentNode[i,k,0,0][0]:
						#	print(sum(poi[j]*CTiltCut[i,k][j] for j in range(len(poi)))+CTiltCut[i,k][-1])
						#print(CTiltCut[i,k])
					if d < Depth[i,k]:
						#print(i,k,d)
						for cutInd in range(len(TiltedCuts[i,k,d])):
							cut = TiltedCuts[i,k,d][cutInd]
							tightInd = (i,k,d,cutInd)
							tight = TightAtCurrentNode[tightInd][0]
							while TightAtCurrentNode[tightInd][1] != None:
								tightInd = TightAtCurrentNode[tightInd][1]
								tight = tight+TightAtCurrentNode[tightInd][0]
							SearchLP = Model()
							SearchLP.params.OutputFlag = 0
							SearchLP.params.threads = 4
							v = SearchLP.addVars(int(self.n/self.r)+1,lb=-1,ub=1)
							w = SearchLP.addVar()
							for kk in range(int(self.n/self.r)):
								if self.w[Sk[k][kk]] > self.c[i]:
									SearchLP.addConstr(v[kk] == 0)
							for poi in tight:
								SearchLP.addConstr(quicksum(poi[ii]*v[ii] for ii in range(int(self.n/self.r)+1)) == w)
							NegVio = SubIPSolns[i,k].dot(cut)
							FoundCand = False
							for kk in range(len(NegVio)):
								if NegVio[-kk] > 1e-6:
									cand = SubIPSolns[i,k][-kk].copy()
									FoundCand = True
									break

							if FoundCand == False:
								subIP[i,k].setObjective(-quicksum(cut[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))-cut[int(self.n/self.r)]*suby[i,k])
								subIP[i,k].update()
								subIP[i,k].optimize()
								nSubIPsolved += 1
								cand = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x,1]

							# Search for tilting direction
							CandCon = SearchLP.addConstr(quicksum(cand[ii]*v[ii] for ii in range(int(self.n/self.r)+1)) == w)
							FoundDir = False
							aim = 0
							sign = 1
							while FoundDir == False:
								# Optimize over direction e_1, e_2,..., e_n, -e_1, -e_2,..., -e_N
								SearchLP.setObjective(sign*v[aim])
								SearchLP.update()
								SearchLP.optimize()
								#print(SearchLP.NumConstrs)
								Dir = [v[i].x for i in range(int(self.n/self.r)+1)]+[w.x]
								if sum(abs(Dir[i]) for i in range(int(self.n/self.r)+1)) <= 1e-8:
									if aim < int(self.n/self.r):
										aim += 1
									else:
										aim = 0
										sign = -1
								else:
									FoundDir = True

							vk1 = [v[ii].x for ii in range(int(self.n/self.r)+1)]
							wk1 = w.x
							vk2 = [-v[ii].x for ii in range(int(self.n/self.r)+1)]
							wk2 = -w.x
							FoundValid = False
							lamk1 = float('inf')
							lamk2 = float('inf')
							new_soln1 = [cand[ii] for ii in range(int(self.n/self.r)+1)]
							new_soln2 = [cand[ii] for ii in range(int(self.n/self.r)+1)]

							# Use known integer solutions to initialize lamk1 and lamk2
							abVio = SubIPSolns[i,k].dot(cut)
							vwVio = SubIPSolns[i,k].dot([v[ii].x for ii in range(int(self.n/self.r)+1)]+[-w.x])
							for kk in range(len(SubIPSolns[i,k])):
								if vwVio[kk] < -1e-8 and lamk1>-abVio[kk]/vwVio[kk]:
									lamk1 = -abVio[kk]/vwVio[kk]
									new_soln1 = [SubIPSolns[i,k][kk][ii] for ii in range(int(self.n/self.r)+1)]
								elif vwVio[kk] > 1e-8 and lamk2>abVio[kk]/vwVio[kk]:
									lamk2 = abVio[kk]/vwVio[kk]
									new_soln2 = [SubIPSolns[i,k][kk][ii] for ii in range(int(self.n/self.r)+1)]
							if lamk1 < float('inf') and lamk1 >= 1e-8:
								vk1 = [cut[ii]+lamk1*v[ii].x for ii in range(int(self.n/self.r)+1)]
								wk1 = -cut[int(self.n/self.r)+1]+lamk1*w.x
							elif lamk1 < 1e-8:
								vk1 = [cut[ii] for ii in range(int(self.n/self.r)+1)]
								wk1 = cut[-1]
							if lamk2 < float('inf') and lamk2 >= 1e-8:
								vk2 = [cut[ii]-lamk2*v[ii].x for ii in range(int(self.n/self.r)+1)]
								wk2 = -cut[int(self.n/self.r)+1]-lamk2*w.x
							elif lamk2 < 1e-8:
								vk2 = [cut[ii] for ii in range(int(self.n/self.r)+1)]
								wk2 = -cut[-1]

							lamk1old = lamk1
							lamk2old = lamk2

							if lamk1 >= 1e-8:
								# Diretion v^Tx >= w
								while FoundValid == False:
									#if (i,k,d) == (3,1,0):
									#	print('lamk1',lamk1)
									if lamk1 < 1e-8 or (lamk1 > lamk1old and lamk1old < 1e-5):
										FoundValid = True
										vk1 = [cut[ii] for ii in range(int(self.n/self.r)+1)]
										wk1 = cut[-1]
										new_soln1 = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]
									else:
										subIP[i,k].setObjective(quicksum(vk1[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+vk1[int(self.n/self.r)]*suby[i,k])
										subIP[i,k].update()
										subIP[i,k].optimize()
										nSubIPsolved += 1
										soln = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x,1.0]
										if soln not in SubIPSolns[i,k].tolist():
											SubIPSolns[i,k]=np.append(SubIPSolns[i,k],[soln],axis=0)
										#SubIPSolns[j]=np.append(SubIPSolns[j],[[subx[j][i].x for i in range(self.N)]+[suby[j].x,1.0]],axis=0)
										if subIP[i,k].objval < wk1-1e-6:
											# vk= a+lamk1*v, wk = b+lamk1*w, lamk1 = (a^Tx^*-b)/(w-v^Tx^*)
											if abs(sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x) < 1e-5:# and subIP[i,k].objval >= wk1-1e-3:
												#print(i,k,d,cutInd,len(TiltedCuts[i,k,d+1]))
												wk1 = subIP[i,k].objval
												FoundValid = True
												break
											lamk1old = lamk1
											lamk1 = (sum(cut[ii]*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*suby[i,k].x+cut[int(self.n/self.r)+1])/(w.x-sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))-v[int(self.n/self.r)].x*suby[i,k].x)
											vk1 = [cut[ii]+lamk1*v[ii].x for ii in range(int(self.n/self.r)+1)]
											wk1 = -cut[int(self.n/self.r)+1]+lamk1*w.x
											new_soln1 = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]
										else:
											wk1 = subIP[i,k].objval
											FoundValid = True
							
							#if i == 0 and k == 2 and d == 4 and cutInd == 13:
							#	subIP[i,k].setObjective(quicksum(cut[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*suby[i,k])
							#	subIP[i,k].update()
							#	subIP[i,k].optimize()
							#	print(subIP[i,k].objval,cut[-1])

							if lamk2 >= 1e-8:
								# Diretion -v^Tx >= -w
								FoundValid = False
								while FoundValid == False:
									#if (i,k,d) == (3,1,0):
									#	print('lamk2',lamk2)
									if lamk2 < 1e-8 or (lamk2 > lamk2old and lamk2old < 1e-5):
										FoundValid = True
										vk2 = [cut[ii] for ii in range(int(self.n/self.r)+1)]
										wk2 = -cut[-1]
										new_soln2 = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]
									else:
										subIP[i,k].setObjective(quicksum(vk2[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+vk2[int(self.n/self.r)]*suby[i,k])
										subIP[i,k].update()
										subIP[i,k].optimize()
										#if i == 0 and k == 2 and d == 4 and cutInd == 13:
										#	print(sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x,\
										#		subIP[i,k].objval,wk2,lamk2,\
										#		sum(cut[ii]*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*suby[i,k].x+cut[int(self.n/self.r)+1],\
										#		sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x)
										nSubIPsolved += 1
										soln = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x,1.0]
										if soln not in SubIPSolns[i,k].tolist():
											SubIPSolns[i,k]=np.append(SubIPSolns[i,k],[soln],axis=0)
										if subIP[i,k].objval < wk2-1e-6:
											# vk= a-lamk2*v, wk = b-lamk2*w, lamk2 = (a^Tx^*-b)/(v^Tx^*-w)
											if abs(sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x) < 1e-5:# and subIP[i,k].objval >= wk2-1e-3:
												#print(i,k,d,cutInd,len(TiltedCuts[i,k,d+1]))
												wk2 = subIP[i,k].objval
												FoundValid = True
												break
											lamk2old = lamk2
											lamk2 = (sum(cut[ii]*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*suby[i,k].x+cut[int(self.n/self.r)+1])/(sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x)
											vk2 = [cut[ii]-lamk2*v[ii].x for ii in range(int(self.n/self.r)+1)]
											wk2 = -cut[int(self.n/self.r)+1]-lamk2*w.x
											new_soln2 = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]
										else:
											wk2 = subIP[i,k].objval
											FoundValid = True

							TiltedCuts[i,k,d+1].append(vk1+[-wk1])
							TightAtCurrentNode[i,k,d+1,len(TiltedCuts[i,k,d+1])-1] = ([new_soln1],(i,k,d,cutInd))
							#subIP[i,k].setObjective(quicksum(vk1[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+vk1[int(self.n/self.r)]*suby[i,k])
							#subIP[i,k].update()
							#subIP[i,k].optimize()
							#if subIP[i,k].objval < wk1-1e-8:
							#	print(d,i,k,lamk1,len(TiltedCuts[i,k,d+1])-1,subIP[i,k].objval,wk1,'*')

							if lamk1 >= 1e-8 or lamk2 >= 1e-8:
								TiltedCuts[i,k,d+1].append(vk2+[-wk2])
								TightAtCurrentNode[i,k,d+1,len(TiltedCuts[i,k,d+1])-1] = ([new_soln2],(i,k,d,cutInd))
								#subIP[i,k].setObjective(quicksum(vk2[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+vk2[int(self.n/self.r)]*suby[i,k])
								#subIP[i,k].update()
								#subIP[i,k].optimize()
								#if subIP[i,k].objval < wk2-1e-8:
								#	print(d,i,k,lamk2,len(TiltedCuts[i,k,d+1])-1,subIP[i,k].objval,wk2,'**')
							
		TiltTime.append(time.time()-t0)

		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,lb=0.0,ub=1.0)
		y = m.addVars(self.m,self.r,lb=0.0,ub=1.0)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.addConstr(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)) >= LB)
		m.update()
		m.optimize()

		constrs = m.getConstrs()
		variables = m.getVars()

		pis = [con.pi for con in constrs]
		rdcosts = [var.RC for var in variables]

		dim_OPTface.append(self.m*self.n+self.m*self.r-len([pi for pi in pis if abs(pi) > 1e-8])-len([rdcost for rdcost in rdcosts if abs(rdcost) > 1e-8]))

		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,lb=0.0,ub=1.0)
		y = m.addVars(self.m,self.r,lb=0.0,ub=1.0)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.addConstrs((-quicksum((self.p[j]+lam_soln[j])*x[i,j] for j in Sk[k])-mu_soln[i]*y[i,k] >= subIPObj[i,k] for i in range(self.m) for k in range(self.r)))
		m.update()
		m.optimize()

		constrs = m.getConstrs()
		variables = m.getVars()

		pis = [con.pi for con in constrs]
		rdcosts = [var.RC for var in variables]

		dim_OPTface.append(self.m*self.n+self.m*self.r-len([pi for pi in pis if abs(pi) > 1e-8])-len([rdcost for rdcost in rdcosts if abs(rdcost) > 1e-8]))

		m = Model()
		#m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,lb=0.0,ub=1.0)
		y = m.addVars(self.m,self.r,lb=0.0,ub=1.0)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.addConstrs((quicksum(CTiltCut[i,k][ii]*x[i,Sk[k][ii]] for ii in range(int(self.n/self.r)))+CTiltCut[i,k][int(self.n/self.r)]*y[i,k] >= -CTiltCut[i,k][int(self.n/self.r)+1] for i in range(self.m) for k in range(self.r)))
		m.update()
		m.optimize()

		constrs = m.getConstrs()
		variables = m.getVars()

		pis = [con.pi for con in constrs]
		rdcosts = [var.RC for var in variables]

		dim_OPTface.append(self.m*self.n+self.m*self.r-len([pi for pi in pis if abs(pi) > 1e-8])-len([rdcost for rdcost in rdcosts if abs(rdcost) > 1e-8]))

		for d in range(1,DepthUB+1):
			m = Model()
			m.params.OutputFlag = 0
			m.params.threads = 4
			x = m.addVars(self.m,self.n,lb=0.0,ub=1.0)
			y = m.addVars(self.m,self.r,lb=0.0,ub=1.0)
			m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
			m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
			m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
			m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
			NewCuts = []
			nCuts = 0
			for i in range(self.m):
				for k in range(self.r):
					for cut in TiltedCuts[i,k,min(d,Depth[i,k])]:
						NewCuts.append(m.addConstr(quicksum(cut[ii]*x[i,Sk[k][ii]] for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*y[i,k] >= -cut[int(self.n/self.r)+1]))
						#NewCuts[nCuts].setAttr('lazy',-1)
						nCuts += 1

			m.update()
			m.optimize()

			constrs = m.getConstrs()
			variables = m.getVars()

			pis = [con.pi for con in constrs]
			rdcosts = [var.RC for var in variables]

			dim_OPTface.append(self.m*self.n+self.m*self.r-len([pi for pi in pis if abs(pi) > 1e-8])-len([rdcost for rdcost in rdcosts if abs(rdcost) > 1e-8]))
			
		print(dim_OPTface)

		OutputStr = ""
		for i in range(len(dim_OPTface)):
			OutputStr = OutputStr+str(dim_OPTface[i])+'\t'
		return OutputStr


	def MultiCoreSim(self,tLimit=10*60):
		Sk = self.Sk
		t0 = time.time()
		t0_Dual = time.time()
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 1
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.params.SolutionLimit = 2
		m.update()
		m.optimize()
		UB = m.objval
		del m

		DualMaster = Model()
		DualMaster.modelSense = GRB.MAXIMIZE
		DualMaster.params.OutputFlag = 0
		DualMaster.params.threads = 1
		lam = DualMaster.addVars(self.n,lb=-float('inf'),ub=0)
		mu = DualMaster.addVars(self.m,lb=-float('inf'),ub=0)
		# theta <= V(lambda)
		theta = DualMaster.addVars(self.m,self.r,lb=-float('inf'))
		DualObj = DualMaster.addVar(lb=-float('inf'),ub=UB)
		DualMaster.addConstr(DualObj == quicksum(lam[j] for j in range(self.n))+quicksum(mu[i] for i in range(self.m))+quicksum(theta[i,k] for i in range(self.m) for k in range(self.r)))
		DualMaster.setObjective(DualObj)
		subIP = {}
		subx = {}
		suby = {}
		for i in range(self.m):
			for k in range(self.r):
				subIP[i,k] = Model()
				subIP[i,k].params.OutputFlag = 0
				subIP[i,k].params.threads = 1
				subx[i,k] = subIP[i,k].addVars(Sk[k],vtype=GRB.BINARY)
				suby[i,k] = subIP[i,k].addVar(vtype=GRB.BINARY)
				subIP[i,k].addConstr(quicksum(self.w[j]*subx[i,k][j] for j in Sk[k]) <= self.c[i]*suby[i,k])
		
		lam_soln = {}
		mu_soln = {}
		LB = -float('inf')
		iterN = 0
		StopCondt = False
		DualMethod = 'level'
		#DualMethod = 'cutpl'
		nSubIPsolved = 0

		DWCuts = {(i,k):np.empty((0,len(Sk[k])+2)) for i in range(self.m) for k in range(self.r)}
		SubIPSolns = {(i,k):np.empty((0,len(Sk[k])+2)) for i in range(self.m) for k in range(self.r)}
		DualTime = 0
		subtime = 0
		LPtime = 0
		t0 = time.time()
		while StopCondt == False:
			iterN += 1
			print(iterN,LB,UB,time.time()-t0)
			DualMaster.update()
			DualMaster.optimize()
			UB = DualMaster.objval
			lam_old = lam_soln.copy()
			mu_old = mu_soln.copy()
			if DualMethod == 'cutpl' or (DualMethod == 'level' and iterN == 1):
				for j in range(self.n):
					lam_soln[j] = lam[j].x
				for i in range(self.m):
					mu_soln[i] = mu[i].x
			elif DualMethod == 'level':
				lt = UB-0.3*(UB-LB)
				levelCon = DualMaster.addConstr(DualObj >= lt)
				DualMaster.setObjective(-quicksum((lam[j]-lam_old[j])*(lam[j]-lam_old[j]) for j in range(self.n))-quicksum((mu[i]-mu_old[i])*(mu[i]-mu_old[i]) for i in range(self.m)))
				DualMaster.optimize()
				if DualMaster.status == 2:
					for j in range(self.n):
						lam_soln[j] = lam[j].x
					for i in range(self.m):
						mu_soln[i] = mu[i].x
				else:
					print('QP solver having numerical issues...')
					break
				DualMaster.remove(levelCon)
				DualMaster.setObjective(DualObj)
			if UB-LB < 1e-6*(min(abs(UB),abs(LB))+1):
				StopCondt = True
				print('Dual problem terminates!')

			# Solve "pricing" subproblem
			for i in range(self.m):
				for k in range(self.r):
					subIP[i,k].setObjective(-quicksum((self.p[j]+lam_soln[j])*subx[i,k][j] for j in Sk[k])-mu_soln[i]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					nSubIPsolved += 1
					if subIP[i,k].status == 2:
						subx_soln = {j:subx[i,k][j].x for j in Sk[k]}
						suby_soln = suby[i,k].x
						DualMaster.addConstr(theta[i,k] <= -sum(self.p[j]*subx_soln[j] for j in Sk[k])-quicksum(subx_soln[j]*lam[j] for j in Sk[k])-suby_soln*mu[i])
						DWCuts[i,k] = np.append(DWCuts[i,k],np.array([[-self.p[j]-lam_soln[j] for j in Sk[k]]+[-mu_soln[i],-subIP[i,k].objval]]),axis=0)
						soln = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x,1.0]
						if soln not in SubIPSolns[i,k].tolist():
							SubIPSolns[i,k]=np.append(SubIPSolns[i,k],[soln],axis=0)
					else:
						print('subIP error!')
			LB_new = sum(lam_soln[j] for j in range(self.n))+sum(mu_soln[i] for i in range(self.m))+sum(subIP[i,k].objval for i in range(self.m) for k in range(self.r))
			if LB_new > LB:
				LB = LB_new
		t_Dual = time.time()-t0_Dual
		subIPObj = {(i,k):subIP[i,k].objval for i in range(self.m) for k in range(self.r)}
		DualUB = UB
		DualLB = LB

		t0_MIP1 = time.time()
		m = Model()
		#m.params.OutputFlag = 0
		m.params.threads = 3
		m.params.TimeLimit = t_Dual
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.update()
		m.optimize()

		xMIP = {(i,j):x[i,j].x for i in range(self.m) for j in range(self.n)}
		yMIP = {(i,k):y[i,k].x for i in range(self.m) for k in range(self.r)}

		t_MIP1 = time.time()-t0_MIP1

		t0_CTilt = time.time()
		DepthUB = 6
		TightAtCurrentNode = {(i,k,0,0):([[subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]],None) for i in range(self.m) for k in range(self.r)}
		
		for i in range(self.m):
			for k in range(self.r):
				subIP[i,k].params.threads = 4

		CTiltCut = {}
		nTilt = 0
		nCSubIPs = 0
		t0_CTilt = time.time()
		for i in range(self.m):
			for k in range(self.r):
				cut = {}
				for j in Sk[k]:
					cut[j] = -self.p[j]-lam_soln[j]
				cut[self.n] = -mu_soln[i]
				cutRHS = subIP[i,k].objval
				CoordStatus = {}
				for j in Sk[k]:
					if subx[i,k][j].x > 0.5:
						CoordStatus[j] = 1
					else:
						CoordStatus[j] = 0
				if suby[i,k].x > 0.5:
					CoordStatus[self.n] = 1
				else:
					CoordStatus[self.n] = 0

				# CoordStatus[j] = 0(/1) means all known feasible points have its j-th coordinate == 0(/1)
				# CoordStatus[j] = 2 means there are known feasible points having its j-th coordinate equal to 0 and equal to 1, in which case one cannot tilt

				for j in Sk[k]:
					if CoordStatus[j] == 1:
						subx[i,k][j].ub = 0.0
						subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
						subIP[i,k].update()
						subIP[i,k].optimize()
						nCSubIPs += 1
						TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
						delta = subIP[i,k].objval - cutRHS
						if delta > 1e-6:
							cutRHS += delta
							cut[j] += delta
							nTilt += 1
						for jj in Sk[k]:
							if CoordStatus[jj] == 0 and subx[i,k][jj].x > 0.5:
								CoordStatus[jj] = 2
							elif CoordStatus[jj] == 1 and subx[i,k][jj].x < 0.5:
								CoordStatus[jj] = 2
						if CoordStatus[self.n] == 0 and suby[i,k].x > 0.5:
							CoordStatus[self.n] = 2
						elif CoordStatus[self.n] == 1 and suby[i,k].x < 0.5:
							CoordStatus[self.n] = 2
						subx[i,k][j].ub = 1.0
						subIP[i,k].update()
					elif CoordStatus[j] == 0 and self.w[j] <= self.c[i]:
						subx[i,k][j].lb = 1.0
						subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
						subIP[i,k].update()
						subIP[i,k].optimize()
						nCSubIPs += 1
						TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
						delta = subIP[i,k].objval - cutRHS
						if delta > 1e-6:
							cut[j] -= delta
							nTilt += 1
						for jj in Sk[k]:
							if CoordStatus[jj] == 0 and subx[i,k][jj].x > 0.5:
								CoordStatus[jj] = 2
							elif CoordStatus[jj] == 1 and subx[i,k][jj].x < 0.5:
								CoordStatus[jj] = 2
						if CoordStatus[self.n] == 0 and suby[i,k].x > 0.5:
							CoordStatus[self.n] = 2
						elif CoordStatus[self.n] == 1 and suby[i,k].x < 0.5:
							CoordStatus[self.n] = 2
						subx[i,k][j].lb = 0.0
						subIP[i,k].update()

				if CoordStatus[self.n] == 1:
					suby[i,k].ub = 0.0
					subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					nCSubIPs += 1
					TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
					delta = subIP[i,k].objval - cutRHS
					if delta > 1e-6:
						cutRHS += delta
						cut[self.n] += delta
						nTilt += 1
					suby[i,k].ub = 1.0
					subIP[i,k].update()
				elif CoordStatus[self.n] == 0:
					suby[i,k].lb = 1.0
					subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					nCSubIPs += 1
					TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
					delta = subIP[i,k].objval - cutRHS
					if delta > 1e-6:
						cut[self.n] -= delta
						nTilt += 1
					suby[i,k].lb = 0.0
					subIP[i,k].update()

				CTiltCut[i,k] = [cut[j] for j in Sk[k]]+[cut[self.n],-cutRHS]
		t_CTilt = time.time()-t0_CTilt

		t0 = time.time()
		TiltTime =[]
		Depth = {(i,k): max(min(DepthUB,len([j for j in Sk[k] if self.w[j] <= self.c[i]])+1-len(TightAtCurrentNode[i,k,0,0][0])),0) for i in range(self.m) for k in range(self.r)}		
		TiltedCuts = {(i,k,d):[] for i in range(self.m) for k in range(self.r) for d in range(Depth[i,k]+1)}
		for i in range(self.m):
			for k in range(self.r):
				TiltedCuts[i,k,0].append(CTiltCut[i,k].copy())
		

		# Tilt the cuts
		for d in range(DepthUB):
			if d >= 1:
				TiltTime.append(time.time()-t0)
			for i in range(self.m):
				for k in range(self.r):
					#if i == 0 and k == 1:
						#print(TightAtCurrentNode[i,k,0,0][0])
						#for poi in TightAtCurrentNode[i,k,0,0][0]:
						#	print(sum(poi[j]*CTiltCut[i,k][j] for j in range(len(poi)))+CTiltCut[i,k][-1])
						#print(CTiltCut[i,k])
					if d < Depth[i,k]:
						#print(i,k,d)
						for cutInd in range(len(TiltedCuts[i,k,d])):
							cut = TiltedCuts[i,k,d][cutInd]
							tightInd = (i,k,d,cutInd)
							tight = TightAtCurrentNode[tightInd][0]
							while TightAtCurrentNode[tightInd][1] != None:
								tightInd = TightAtCurrentNode[tightInd][1]
								tight = tight+TightAtCurrentNode[tightInd][0]
							SearchLP = Model()
							SearchLP.params.OutputFlag = 0
							SearchLP.params.threads = 4
							v = SearchLP.addVars(int(self.n/self.r)+1,lb=-1,ub=1)
							w = SearchLP.addVar()
							for kk in range(int(self.n/self.r)):
								if self.w[Sk[k][kk]] > self.c[i]:
									SearchLP.addConstr(v[kk] == 0)
							for poi in tight:
								SearchLP.addConstr(quicksum(poi[ii]*v[ii] for ii in range(int(self.n/self.r)+1)) == w)
							NegVio = SubIPSolns[i,k].dot(cut)
							FoundCand = False
							for kk in range(len(NegVio)):
								if NegVio[-kk] > 1e-6:
									cand = SubIPSolns[i,k][-kk].copy()
									FoundCand = True
									break

							if FoundCand == False:
								subIP[i,k].setObjective(-quicksum(cut[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))-cut[int(self.n/self.r)]*suby[i,k])
								subIP[i,k].update()
								subIP[i,k].optimize()
								nSubIPsolved += 1
								cand = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x,1]

							# Search for tilting direction
							CandCon = SearchLP.addConstr(quicksum(cand[ii]*v[ii] for ii in range(int(self.n/self.r)+1)) == w)
							FoundDir = False
							aim = 0
							sign = 1
							while FoundDir == False:
								# Optimize over direction e_1, e_2,..., e_n, -e_1, -e_2,..., -e_N
								SearchLP.setObjective(sign*v[aim])
								SearchLP.update()
								SearchLP.optimize()
								#print(SearchLP.NumConstrs)
								Dir = [v[i].x for i in range(int(self.n/self.r)+1)]+[w.x]
								if sum(abs(Dir[i]) for i in range(int(self.n/self.r)+1)) <= 1e-8:
									if aim < int(self.n/self.r):
										aim += 1
									else:
										aim = 0
										sign = -1
								else:
									FoundDir = True

							vk1 = [v[ii].x for ii in range(int(self.n/self.r)+1)]
							wk1 = w.x
							vk2 = [-v[ii].x for ii in range(int(self.n/self.r)+1)]
							wk2 = -w.x
							FoundValid = False
							lamk1 = float('inf')
							lamk2 = float('inf')
							new_soln1 = [cand[ii] for ii in range(int(self.n/self.r)+1)]
							new_soln2 = [cand[ii] for ii in range(int(self.n/self.r)+1)]

							# Use known integer solutions to initialize lamk1 and lamk2
							abVio = SubIPSolns[i,k].dot(cut)
							vwVio = SubIPSolns[i,k].dot([v[ii].x for ii in range(int(self.n/self.r)+1)]+[-w.x])
							for kk in range(len(SubIPSolns[i,k])):
								if vwVio[kk] < -1e-8 and lamk1>-abVio[kk]/vwVio[kk]:
									lamk1 = -abVio[kk]/vwVio[kk]
									new_soln1 = [SubIPSolns[i,k][kk][ii] for ii in range(int(self.n/self.r)+1)]
								elif vwVio[kk] > 1e-8 and lamk2>abVio[kk]/vwVio[kk]:
									lamk2 = abVio[kk]/vwVio[kk]
									new_soln2 = [SubIPSolns[i,k][kk][ii] for ii in range(int(self.n/self.r)+1)]
							if lamk1 < float('inf') and lamk1 >= 1e-8:
								vk1 = [cut[ii]+lamk1*v[ii].x for ii in range(int(self.n/self.r)+1)]
								wk1 = -cut[int(self.n/self.r)+1]+lamk1*w.x
							elif lamk1 < 1e-8:
								vk1 = [cut[ii] for ii in range(int(self.n/self.r)+1)]
								wk1 = cut[-1]
							if lamk2 < float('inf') and lamk2 >= 1e-8:
								vk2 = [cut[ii]-lamk2*v[ii].x for ii in range(int(self.n/self.r)+1)]
								wk2 = -cut[int(self.n/self.r)+1]-lamk2*w.x
							elif lamk2 < 1e-8:
								vk2 = [cut[ii] for ii in range(int(self.n/self.r)+1)]
								wk2 = -cut[-1]

							lamk1old = lamk1
							lamk2old = lamk2

							if lamk1 >= 1e-8:
								# Diretion v^Tx >= w
								while FoundValid == False:
									#print('lamk1',lamk1)
									if lamk1 < 1e-8 or (lamk1 > lamk1old and lamk1old < 1e-5):
										FoundValid = True
										vk1 = [cut[ii] for ii in range(int(self.n/self.r)+1)]
										wk1 = cut[-1]
										new_soln1 = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]
									else:
										subIP[i,k].setObjective(quicksum(vk1[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+vk1[int(self.n/self.r)]*suby[i,k])
										subIP[i,k].update()
										subIP[i,k].optimize()
										nSubIPsolved += 1
										soln = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x,1.0]
										if soln not in SubIPSolns[i,k].tolist():
											SubIPSolns[i,k]=np.append(SubIPSolns[i,k],[soln],axis=0)
										#SubIPSolns[j]=np.append(SubIPSolns[j],[[subx[j][i].x for i in range(self.N)]+[suby[j].x,1.0]],axis=0)
										if subIP[i,k].objval < wk1-1e-6:
											# vk= a+lamk1*v, wk = b+lamk1*w, lamk1 = (a^Tx^*-b)/(w-v^Tx^*)
											if abs(sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x) < 1e-5 and subIP[i,k].objval >= wk1-1e-3:
												#print(i,k,d,cutInd,len(TiltedCuts[i,k,d+1]))
												wk1 = subIP[i,k].objval
												FoundValid = True
												break
											lamk1old = lamk1
											lamk1 = (sum(cut[ii]*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*suby[i,k].x+cut[int(self.n/self.r)+1])/(w.x-sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))-v[int(self.n/self.r)].x*suby[i,k].x)
											vk1 = [cut[ii]+lamk1*v[ii].x for ii in range(int(self.n/self.r)+1)]
											wk1 = -cut[int(self.n/self.r)+1]+lamk1*w.x
											new_soln1 = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]
										else:
											wk1 = subIP[i,k].objval
											FoundValid = True
							
							#if i == 0 and k == 2 and d == 4 and cutInd == 13:
							#	subIP[i,k].setObjective(quicksum(cut[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*suby[i,k])
							#	subIP[i,k].update()
							#	subIP[i,k].optimize()
							#	print(subIP[i,k].objval,cut[-1])

							if lamk2 >= 1e-8:
								# Diretion -v^Tx >= -w
								FoundValid = False
								while FoundValid == False:
									if lamk2 < 1e-8 or (lamk2 > lamk2old and lamk2old < 1e-5):
										FoundValid = True
										vk2 = [cut[ii] for ii in range(int(self.n/self.r)+1)]
										wk2 = -cut[-1]
										new_soln2 = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]
									else:
										subIP[i,k].setObjective(quicksum(vk2[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+vk2[int(self.n/self.r)]*suby[i,k])
										subIP[i,k].update()
										subIP[i,k].optimize()
										#if i == 0 and k == 2 and d == 4 and cutInd == 13:
										#	print(sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x,\
										#		subIP[i,k].objval,wk2,lamk2,\
										#		sum(cut[ii]*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*suby[i,k].x+cut[int(self.n/self.r)+1],\
										#		sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x)
										nSubIPsolved += 1
										soln = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x,1.0]
										if soln not in SubIPSolns[i,k].tolist():
											SubIPSolns[i,k]=np.append(SubIPSolns[i,k],[soln],axis=0)
										if subIP[i,k].objval < wk2-1e-6:
											# vk= a-lamk2*v, wk = b-lamk2*w, lamk2 = (a^Tx^*-b)/(v^Tx^*-w)
											if abs(sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x) < 1e-5 and subIP[i,k].objval >= wk2-1e-3:
												#print(i,k,d,cutInd,len(TiltedCuts[i,k,d+1]))
												wk2 = subIP[i,k].objval
												FoundValid = True
												break
											lamk2old = lamk2
											lamk2 = (sum(cut[ii]*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*suby[i,k].x+cut[int(self.n/self.r)+1])/(sum(v[ii].x*subx[i,k][Sk[k][ii]].x for ii in range(int(self.n/self.r)))+v[int(self.n/self.r)].x*suby[i,k].x-w.x)
											vk2 = [cut[ii]-lamk2*v[ii].x for ii in range(int(self.n/self.r)+1)]
											wk2 = -cut[int(self.n/self.r)+1]-lamk2*w.x
											new_soln2 = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]
										else:
											wk2 = subIP[i,k].objval
											FoundValid = True

							TiltedCuts[i,k,d+1].append(vk1+[-wk1])
							TightAtCurrentNode[i,k,d+1,len(TiltedCuts[i,k,d+1])-1] = ([new_soln1],(i,k,d,cutInd))
							#subIP[i,k].setObjective(quicksum(vk1[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+vk1[int(self.n/self.r)]*suby[i,k])
							#subIP[i,k].update()
							#subIP[i,k].optimize()
							#if subIP[i,k].objval < wk1-1e-8:
							#	print(d,i,k,lamk1,len(TiltedCuts[i,k,d+1])-1,subIP[i,k].objval,wk1,'*')

							if lamk1 >= 1e-8 or lamk2 >= 1e-8:
								TiltedCuts[i,k,d+1].append(vk2+[-wk2])
								TightAtCurrentNode[i,k,d+1,len(TiltedCuts[i,k,d+1])-1] = ([new_soln2],(i,k,d,cutInd))
								#subIP[i,k].setObjective(quicksum(vk2[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+vk2[int(self.n/self.r)]*suby[i,k])
								#subIP[i,k].update()
								#subIP[i,k].optimize()
								#if subIP[i,k].objval < wk2-1e-8:
								#	print(d,i,k,lamk2,len(TiltedCuts[i,k,d+1])-1,subIP[i,k].objval,wk2,'**')
							
		TiltTime.append(time.time()-t0)


		d = 6
		m = Model()
		#m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		NewCuts = []
		nCuts = 0
		for i in range(self.m):
			for k in range(self.r):
				for cut in TiltedCuts[i,k,min(d,Depth[i,k])]:
					#NewCuts.append(m.addConstr(quicksum(cut[ii]*x[i,Sk[k][ii]] for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*y[i,k] >= -cut[int(self.n/self.r)+1]))
					NewCuts.append(m.addConstr(quicksum(cut[ii]*x[i,Sk[k][ii]] for ii in range(int(self.n/self.r)))+cut[int(self.n/self.r)]*y[i,k] >= -cut[int(self.n/self.r)+1]-1e-5))
					#NewCuts[nCuts].setAttr('lazy',-1)
					nCuts += 1
		for i in range(self.m):
			for j in range(self.n):
				x[i,j].Start = xMIP[i,j]
		for i in range(self.m):
			for k in range(self.r):
				y[i,k].Start = yMIP[i,k]
		m.params.TimeLimit = tLimit-t_MIP1-t_CTilt-TiltTime[-1]
		m.params.LogFile = "LogFiles/MKAP/"+self.InstName+"_DWcutsCT"+str(d)+"_MultiCoreSimulation_"+str(tLimit)+".log"
		m.update()
		t0 = time.time()
		m.optimize()
		t_MIPT6 = time.time()-t0

		if m.status == 2:
			ifSolveCTd = 1
		else:
			ifSolveCTd = 0
		nNodesCTd = m.NodeCount
		GapCTd = m.MIPGap
		LBCTd = m.ObjBound
		UBCTd = m.objval

		Output = [t_Dual,t_MIP1,t_CTilt,TiltTime[-1],t_MIPT6,ifSolveCTd,nNodesCTd,GapCTd,LBCTd,UBCTd]
		
		OutputStr = ""
		for i in range(len(Output)):
			OutputStr = OutputStr+str(Output[i])+'\t'
		return OutputStr

	def Labeling(self,tLimit=10*60):
		Sk = self.Sk
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.update()
		m_LP = m.relax()
		m_LP.update()
		m_LP.optimize()
		LB_LP = m_LP.ObjBound
		m.params.NodeLimit = 1
		m.update()
		t0 = time.time()
		m.optimize()
		RootTime0 = time.time()-t0
		LB_Root = m.ObjBound
		del m
		del m_LP

		t0 = time.time()
		t0_Dual = time.time()
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 1
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.params.SolutionLimit = 2
		m.update()
		m.optimize()
		UB = m.objval
		del m

		DualMaster = Model()
		DualMaster.modelSense = GRB.MAXIMIZE
		DualMaster.params.OutputFlag = 0
		DualMaster.params.threads = 1
		lam = DualMaster.addVars(self.n,lb=-float('inf'),ub=0)
		mu = DualMaster.addVars(self.m,lb=-float('inf'),ub=0)
		# theta <= V(lambda)
		theta = DualMaster.addVars(self.m,self.r,lb=-float('inf'))
		DualObj = DualMaster.addVar(lb=-float('inf'),ub=UB)
		DualMaster.addConstr(DualObj == quicksum(lam[j] for j in range(self.n))+quicksum(mu[i] for i in range(self.m))+quicksum(theta[i,k] for i in range(self.m) for k in range(self.r)))
		DualMaster.setObjective(DualObj)
		subIP = {}
		subx = {}
		suby = {}
		for i in range(self.m):
			for k in range(self.r):
				subIP[i,k] = Model()
				subIP[i,k].params.OutputFlag = 0
				subIP[i,k].params.threads = 1
				subx[i,k] = subIP[i,k].addVars(Sk[k],vtype=GRB.BINARY)
				suby[i,k] = subIP[i,k].addVar(vtype=GRB.BINARY)
				subIP[i,k].addConstr(quicksum(self.w[j]*subx[i,k][j] for j in Sk[k]) <= self.c[i]*suby[i,k])
		
		lam_soln = {}
		mu_soln = {}
		LB = -float('inf')
		iterN = 0
		StopCondt = False
		DualMethod = 'level'
		#DualMethod = 'cutpl'
		nSubIPsolved = 0

		DWCuts = {(i,k):np.empty((0,len(Sk[k])+2)) for i in range(self.m) for k in range(self.r)}
		SubIPSolns = {(i,k):np.empty((0,len(Sk[k])+2)) for i in range(self.m) for k in range(self.r)}
		DualTime = 0
		subtime = 0
		LPtime = 0
		t0 = time.time()
		while StopCondt == False:
			iterN += 1
			print(iterN,LB,UB,time.time()-t0)
			DualMaster.update()
			DualMaster.optimize()
			UB = DualMaster.objval
			lam_old = lam_soln.copy()
			mu_old = mu_soln.copy()
			if DualMethod == 'cutpl' or (DualMethod == 'level' and iterN == 1):
				for j in range(self.n):
					lam_soln[j] = lam[j].x
				for i in range(self.m):
					mu_soln[i] = mu[i].x
			elif DualMethod == 'level':
				lt = UB-0.3*(UB-LB)
				levelCon = DualMaster.addConstr(DualObj >= lt)
				DualMaster.setObjective(-quicksum((lam[j]-lam_old[j])*(lam[j]-lam_old[j]) for j in range(self.n))-quicksum((mu[i]-mu_old[i])*(mu[i]-mu_old[i]) for i in range(self.m)))
				DualMaster.optimize()
				if DualMaster.status == 2:
					for j in range(self.n):
						lam_soln[j] = lam[j].x
					for i in range(self.m):
						mu_soln[i] = mu[i].x
				else:
					print('QP solver having numerical issues...')
					break
				DualMaster.remove(levelCon)
				DualMaster.setObjective(DualObj)
			if UB-LB < 1e-6*(min(abs(UB),abs(LB))+1):
				StopCondt = True
				print('Dual problem terminates!')

			# Solve "pricing" subproblem
			for i in range(self.m):
				for k in range(self.r):
					subIP[i,k].setObjective(-quicksum((self.p[j]+lam_soln[j])*subx[i,k][j] for j in Sk[k])-mu_soln[i]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					nSubIPsolved += 1
					if subIP[i,k].status == 2:
						subx_soln = {j:subx[i,k][j].x for j in Sk[k]}
						suby_soln = suby[i,k].x
						DualMaster.addConstr(theta[i,k] <= -sum(self.p[j]*subx_soln[j] for j in Sk[k])-quicksum(subx_soln[j]*lam[j] for j in Sk[k])-suby_soln*mu[i])
						DWCuts[i,k] = np.append(DWCuts[i,k],np.array([[-self.p[j]-lam_soln[j] for j in Sk[k]]+[-mu_soln[i],-subIP[i,k].objval]]),axis=0)
						soln = [subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x,1.0]
						if soln not in SubIPSolns[i,k].tolist():
							SubIPSolns[i,k]=np.append(SubIPSolns[i,k],[soln],axis=0)
					else:
						print('subIP error!')
			LB_new = sum(lam_soln[j] for j in range(self.n))+sum(mu_soln[i] for i in range(self.m))+sum(subIP[i,k].objval for i in range(self.m) for k in range(self.r))
			if LB_new > LB:
				LB = LB_new
		t_Dual = time.time()-t0_Dual
		subIPObj = {(i,k):subIP[i,k].objval for i in range(self.m) for k in range(self.r)}
		DualUB = UB
		DualLB = LB

		t0_MIP1 = time.time()
		m = Model()
		#m.params.OutputFlag = 0
		m.params.LogFile = "LogFiles/MKAP/"+self.InstName+"_NoSwitch.log"
		m.params.threads = 3
		m.params.TimeLimit = t_Dual
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.update()
		m.optimize()

		MIP1SolCount = m.SolCount

		if MIP1SolCount > 0:
			xMIP = {(i,j):x[i,j].x for i in range(self.m) for j in range(self.n)}
			yMIP = {(i,k):y[i,k].x for i in range(self.m) for k in range(self.r)}
			zLB1 = m.ObjBound
			zUB1 = m.objval
			Gap1 = m.MIPGap
		if m.status == 2:
			MIPsolvedbeforeDW = 1
		else:
			MIPsolvedbeforeDW = 0

		t_MIP1 = time.time()-t0_MIP1

		t0_MIP2 = time.time()
		m.params.threads = 4
		m.params.TimeLimit = tLimit - t_MIP1
		m.update()
		m.optimize()
		zLB2 = m.ObjBound
		zUB2 = m.objval
		nNodes2 = m.NodeCount
		Gap2 = m.MIPGap
		if m.status == 2:
			MIP2solved = 1
		else:
			MIP2solved = 0
		t_MIP2 = time.time()-t0_MIP2
		t_NoSwitchTotal = t_MIP1+t_MIP2

		t0_CTilt = time.time()
		DepthUB = 6
		TightAtCurrentNode = {(i,k,0,0):([[subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x]],None) for i in range(self.m) for k in range(self.r)}
		
		for i in range(self.m):
			for k in range(self.r):
				subIP[i,k].params.threads = 4

		CTiltCut = {}
		nTilt = 0
		nCSubIPs = 0
		for i in range(self.m):
			for k in range(self.r):
				cut = {}
				for j in Sk[k]:
					cut[j] = -self.p[j]-lam_soln[j]
				cut[self.n] = -mu_soln[i]
				cutRHS = subIP[i,k].objval
				CoordStatus = {}
				for j in Sk[k]:
					if subx[i,k][j].x > 0.5:
						CoordStatus[j] = 1
					else:
						CoordStatus[j] = 0
				if suby[i,k].x > 0.5:
					CoordStatus[self.n] = 1
				else:
					CoordStatus[self.n] = 0

				# CoordStatus[j] = 0(/1) means all known feasible points have its j-th coordinate == 0(/1)
				# CoordStatus[j] = 2 means there are known feasible points having its j-th coordinate equal to 0 and equal to 1, in which case one cannot tilt

				for j in Sk[k]:
					if CoordStatus[j] == 1:
						subx[i,k][j].ub = 0.0
						subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
						subIP[i,k].update()
						subIP[i,k].optimize()
						nCSubIPs += 1
						TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
						delta = subIP[i,k].objval - cutRHS
						if delta > 1e-6:
							cutRHS += delta
							cut[j] += delta
							nTilt += 1
						for jj in Sk[k]:
							if CoordStatus[jj] == 0 and subx[i,k][jj].x > 0.5:
								CoordStatus[jj] = 2
							elif CoordStatus[jj] == 1 and subx[i,k][jj].x < 0.5:
								CoordStatus[jj] = 2
						if CoordStatus[self.n] == 0 and suby[i,k].x > 0.5:
							CoordStatus[self.n] = 2
						elif CoordStatus[self.n] == 1 and suby[i,k].x < 0.5:
							CoordStatus[self.n] = 2
						subx[i,k][j].ub = 1.0
						subIP[i,k].update()
					elif CoordStatus[j] == 0 and self.w[j] <= self.c[i]:
						subx[i,k][j].lb = 1.0
						subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
						subIP[i,k].update()
						subIP[i,k].optimize()
						nCSubIPs += 1
						TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
						delta = subIP[i,k].objval - cutRHS
						if delta > 1e-6:
							cut[j] -= delta
							nTilt += 1
						for jj in Sk[k]:
							if CoordStatus[jj] == 0 and subx[i,k][jj].x > 0.5:
								CoordStatus[jj] = 2
							elif CoordStatus[jj] == 1 and subx[i,k][jj].x < 0.5:
								CoordStatus[jj] = 2
						if CoordStatus[self.n] == 0 and suby[i,k].x > 0.5:
							CoordStatus[self.n] = 2
						elif CoordStatus[self.n] == 1 and suby[i,k].x < 0.5:
							CoordStatus[self.n] = 2
						subx[i,k][j].lb = 0.0
						subIP[i,k].update()

				if CoordStatus[self.n] == 1:
					suby[i,k].ub = 0.0
					subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					nCSubIPs += 1
					TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
					delta = subIP[i,k].objval - cutRHS
					if delta > 1e-6:
						cutRHS += delta
						cut[self.n] += delta
						nTilt += 1
					suby[i,k].ub = 1.0
					subIP[i,k].update()
				elif CoordStatus[self.n] == 0:
					suby[i,k].lb = 1.0
					subIP[i,k].setObjective(quicksum(cut[j]*subx[i,k][j] for j in Sk[k])+cut[self.n]*suby[i,k])
					subIP[i,k].update()
					subIP[i,k].optimize()
					nCSubIPs += 1
					TightAtCurrentNode[i,k,0,0][0].append([subx[i,k][j].x for j in Sk[k]]+[suby[i,k].x])
					delta = subIP[i,k].objval - cutRHS
					if delta > 1e-6:
						cut[self.n] -= delta
						nTilt += 1
					suby[i,k].lb = 0.0
					subIP[i,k].update()

				CTiltCut[i,k] = [cut[j] for j in Sk[k]]+[cut[self.n],-cutRHS]
		t_CTilt = time.time()-t0_CTilt

		m = Model()
		#m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.m,self.n,vtype=GRB.BINARY)
		y = m.addVars(self.m,self.r,vtype=GRB.BINARY)
		m.setObjective(quicksum(-self.p[j]*x[i,j] for j in range(self.n) for i in range(self.m)))
		m.addConstrs((quicksum(x[i,j] for i in range(self.m)) <= 1 for j in range(self.n)))
		m.addConstrs((quicksum(self.w[j]*x[i,j] for j in Sk[k]) <= self.c[i]*y[i,k] for i in range(self.m) for k in range(self.r)))
		m.addConstrs((quicksum(y[i,k] for k in range(self.r)) <= 1 for i in range(self.m)))
		m.addConstrs((quicksum(CTiltCut[i,k][ii]*x[i,Sk[k][ii]] for ii in range(int(self.n/self.r)))+CTiltCut[i,k][int(self.n/self.r)]*y[i,k] >= -CTiltCut[i,k][int(self.n/self.r)+1]-1e-5 for i in range(self.m) for k in range(self.r)))
		if MIP1SolCount > 0:
			for i in range(self.m):
				for j in range(self.n):
					x[i,j].Start = xMIP[i,j]
			for i in range(self.m):
				for k in range(self.r):
					y[i,k].Start = yMIP[i,k]
		m.params.TimeLimit = tLimit - t_MIP1 - t_CTilt
		m.params.LogFile = "LogFiles/MKAP/"+self.InstName+"_Switch.log"
		m.update()
		t0 = time.time()
		m.optimize()
		t_IPCT = time.time()-t0
		t_SwitchTotal = t_MIP1+t_CTilt+t_IPCT
		if m.status == 2:
			ifSolveCT = 1
		else:
			ifSolveCT = 0
		nNodesCT = m.NodeCount
		GapCT = m.MIPGap
		LBCT = m.ObjBound
		UBCT = m.objval
		del m

		if MIPsolvedbeforeDW == 1:
			label = None
		else:
			if ifSolveCT == 1 and MIP2solved == 0:
				label = 1
			elif ifSolveCT == 0 and MIP2solved == 1:
				label = 0
			elif ifSolveCT == 0 and MIP2solved == 0:
				if GapCT < Gap2:
					label = 1
				else:
					label = 0
			elif ifSolveCT == 1 and MIP2solved == 1:
				if t_SwitchTotal < 0.9*t_NoSwitchTotal:
					label = 1
				else:
					label = 0

		Output = [label,LB_LP,LB_Root,DualLB,zLB1,zUB1,t_Dual,t_MIP1,t_CTilt,t_MIP2,t_IPCT,t_SwitchTotal,t_NoSwitchTotal,Gap1,Gap2,GapCT,MIPsolvedbeforeDW,ifSolveCT,MIP2solved]
		
		OutputStr = ""
		for i in range(len(Output)):
			OutputStr = OutputStr+str(Output[i])+'\t'
		return OutputStr



'''
for r,m,n in [2,20,50],[2,30,50],[2,40,50],[5,10,50],[5,20,50],[5,30,50],[5,40,50],[10,10,50],[10,20,50],[10,30,50],[10,40,50],[10,40,100],[25,10,50],[25,20,50],[25,20,100],\
	[25,30,50],[25,30,100],[25,30,200],[25,40,50],[25,40,100]:#[[10,10,100],[10,10,200],[10,10,300],[25,10,100],[25,10,200],[25,10,300],[25,20,200],[25,20,300],[25,30,300]]:
	for inst in range(1,11):
		for InstType in ['0U','1W','2S']:
			ss = MKAP(m,n,r,inst,InstType)
			ss.ReadNew()
			OutputStr = ss.ExtCompare()
			WrtStr = InstType+'\t'+str(m)+'\t'+str(n)+'\t'+str(r)+'\t'+str(inst)+'\t'+OutputStr+'\n'
			f = open('MKAP_ROOTgood_ExtCompare600.txt','a')#'MKAP_DWgood_ExtCompare600.txt','a')
			f.write(WrtStr)
			f.close()
'''

'''
for r,m,n in [[2,20,50],[2,30,50],[2,40,50],[5,10,50],[5,20,50],[5,30,50],[5,40,50],[10,10,50],[10,10,100],[10,10,200],[10,10,300],[10,20,50],[10,30,50],[10,40,50],\
	[10,40,100],[25,10,50],[25,10,100],[25,10,200],[25,10,300],[25,20,50],[25,20,100],[25,20,200],[25,20,300],[25,30,50],[25,30,100],[25,30,200],[25,30,300],\
	[25,40,50],[25,40,100]]:
	for inst in range(1,11):
		for InstType in ['0U','1W','2S']:
			ss = MKAP(m,n,r,inst,InstType)
			ss.ReadNew()
			OutputStr = ss.ComputeDim()
			WrtStr = InstType+'\t'+str(m)+'\t'+str(n)+'\t'+str(r)+'\t'+str(inst)+'\t'+OutputStr+'\n'
			f = open('MKAP_NewCompareDim.txt','a')
			f.write(WrtStr)
			f.close()
'''

'''
for r,m,n in [[10,10,100],[10,10,200],[10,10,300],[25,10,100],[25,10,200],[25,10,300],[25,20,200],[25,20,300],[25,30,300]]:
	for inst in range(1,11):
		for InstType in ['0U','1W','2S']:
			ss = MKAP(m,n,r,inst,InstType)
			ss.ReadNew()
			OutputStr = ss.CompareCTilt()
			WrtStr = InstType+'\t'+str(m)+'\t'+str(n)+'\t'+str(r)+'\t'+str(inst)+'\t'+OutputStr+'\n'
			f = open('MKAP_DWgood_CompareTiltp600.txt','a')
			f.write(WrtStr)
			f.close()

for r,m,n in [2,20,50],[2,30,50],[2,40,50],[5,10,50],[5,20,50],[5,30,50],[5,40,50],[10,10,50],[10,20,50],[10,30,50],[10,40,50],[10,40,100],[25,10,50],[25,20,50],[25,20,100],\
	[25,30,50],[25,30,100],[25,30,200],[25,40,50],[25,40,100]:#[[10,10,100],[10,10,200],[10,10,300],[25,10,100],[25,10,200],[25,10,300],[25,20,200],[25,20,300],[25,30,300]]:
	for inst in range(1,11):
		for InstType in ['0U','1W','2S']:
			ss = MKAP(m,n,r,inst,InstType)
			ss.ReadNew()
			OutputStr = ss.CompareCTilt()
			WrtStr = InstType+'\t'+str(m)+'\t'+str(n)+'\t'+str(r)+'\t'+str(inst)+'\t'+OutputStr+'\n'
			f = open('MKAP_ROOTgood_CompareTiltp600.txt','a')#'MKAP_DWgood_CompareTilt600_1.txt','a')
			f.write(WrtStr)
			f.close()
'''

'''
r,m,n = 25,30,300
inst = 4
InstType = '0U'
ss = MKAP(m,n,r,inst,InstType)
ss.ReadNew()
ss.Debug()
'''

'''
r,m,n = 25,30,300
for inst in range(1,11):
	for InstType in ['0U','1W','2S']:
		ss = MKAP(m,n,r,inst,InstType)
		ss.ReadNew()
		OutputStr = ss.MultiCoreSim()
		WrtStr = InstType+'\t'+str(m)+'\t'+str(n)+'\t'+str(r)+'\t'+str(inst)+'\t'+OutputStr+'\n'
		f = open('MKAP_MultiCoreSimulation_600.txt','a')
		f.write(WrtStr)
		f.close()
'''

'''
for r,m,n in [2,20,50],[2,30,50],[2,40,50],[5,10,50],[5,20,50],[5,30,50],[5,40,50],[10,10,50],[10,10,100],[10,10,200],[10,10,300],[10,20,50],[10,30,50],[10,40,50],[10,40,100],\
	[25,10,50],[25,10,100],[25,10,200],[25,10,300],[25,20,50],[25,20,100],[25,20,200],[25,20,300],[25,30,50],[25,30,100],[25,30,200],[25,30,300],[25,40,50],[25,40,100]:
	for InstType in ['0U','1W','2S']:
		for inst in range(1,11):
			ss = MKAP(m,n,r,inst,InstType)
			ss.ReadNew()
			OutputStr = ss.Labeling()
			WrtStr = InstType+'\t'+str(m)+'\t'+str(n)+'\t'+str(r)+'\t'+str(inst)+'\t'+OutputStr+'\n'
			f = open('MKAP_Label.txt','a')
			f.write(WrtStr)
			f.close()
'''
'''
for r in [2,5,10,25]:
	for m in [10,20,30,40]:
		for n in [50,100,200,300]:
			for InstType in ['0U','1W','2S']:
				for inst in range(1,11):
					ss = MKAP(m,n,r,inst,InstType)
					ss.ReadNew()
					ss.CompareDWLevel()
'''