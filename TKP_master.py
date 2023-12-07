from gurobipy import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import time
import math
import random

class TKP:
	def __init__(self,instanceName,Qsize):
		self.instanceName = instanceName
		self.n = None
		self.C = None
		self.Qsize = Qsize
		self.Qlen = None
		self.p = []
		self.S = []
		self.s = []
		self.e = []
		self.w = []
		self.Q = []
		self.xind = []
	def readInstance(self):
		instanceFile = "TKP_instances/"+self.instanceName
		f = open(instanceFile)
		x = f.readlines()
		f.close()
		self.n = int(x[0].strip('\n'))
		self.C = int(x[1].strip('\n'))
		for i in range(2,self.n+2):
			lineinfo = x[i].strip('\n').split(' ')
			self.p.append(float(lineinfo[0]))
			self.w.append(float(lineinfo[1]))
			self.s.append(int(lineinfo[2]))
			self.e.append(int(lineinfo[3]))
		for j in range(self.n):
			'''
			Sj = [i for i in range(self.n) if self.s[i]<=self.s[j] and self.e[i]>self.s[j]]
			if j == 0 or self.S[-1] != Sj:
				self.S.append(Sj)
			'''
			self.S.append([i for i in range(self.n) if self.s[i]<=self.s[j] and self.e[i]>self.s[j]])
		self.Qlen = math.ceil(self.n/self.Qsize)
		print("Reading instance "+self.instanceName+" with S="+str(self.Qsize)+", number of blocks: "+str(self.Qlen))
		for k in range(self.Qlen):
			self.Q.append(range(self.Qsize*k,min(self.Qsize*(k+1),self.n)))
			self.xind.append([])
			for j in self.Q[k]:
				for i in self.S[j]:
					if i not in self.xind[k]:
						self.xind[k].append(i)

	def ExtCompare(self,tLimit=60*60):
		# Initialize a UB
		t0_Dual = time.time()
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.n,vtype=GRB.BINARY)
		m.setObjective(-quicksum(self.p[i]*x[i] for i in range(self.n)))
		m.addConstrs((quicksum(self.w[i]*x[i] for i in self.S[j])<= self.C for j in range(self.n)))
		m.params.SolutionLimit = 2
		m.update()
		m.optimize()
		UB = m.objval
		del m

		# Solve the dual model by cutpl/level
		DualMaster = Model()
		DualMaster.params.OutputFlag = 0
		DualMaster.modelSense = GRB.MAXIMIZE
		DualMaster.params.threads = 4
		mu = {}
		for j in range(self.Qlen):
			for i in self.xind[j]:
				mu[j,i] = DualMaster.addVar(lb=-float('inf'),ub=0.0)
		for i in range(self.n):
			#print([j for j in range(self.Qlen) if i in self.xind[j]])
			DualMaster.addConstr(quicksum(mu[j,i] for j in range(self.Qlen) if i in self.xind[j])<=-self.p[i])
		theta = DualMaster.addVars(self.Qlen,lb=-float('inf'),ub=0.0)
		#DualObj = DualMaster.addVar(lb=-float('inf'),ub=UB)
		DualObj = DualMaster.addVar(lb=-float('inf'))
		DualMaster.addConstr(DualObj==quicksum(theta[j] for j in range(self.Qlen)))
		DualMaster.setObjective(DualObj)

		subIP = {}
		subx = {}
		for j in range(self.Qlen):
			subIP[j] = Model()
			subIP[j].params.OutputFlag = 0
			subIP[j].params.threads = 4
			subIP[j].params.MIPGap = 0
			subx[j] = subIP[j].addVars(self.xind[j],vtype=GRB.BINARY)
			for k in self.Q[j]:
				subIP[j].addConstr(quicksum(self.w[i]*subx[j][i] for i in self.S[k]) <= self.C)

		mu_soln = {}
		LB = -float('inf')
		iterN = 0
		StopCondt = False
		DualMethod = 'level'
		#DualMethod = 'cutpl'
		nSubIPsolved = 0

		DWCuts = {j:np.empty((0,len(self.xind[j])+1)) for j in range(self.Qlen)}
		SubIPSolns = {j:np.empty((0,len(self.xind[j])+1)) for j in range(self.Qlen)}
		DualTime = 0
		subtime = 0
		LPtime = 0
		t0 = time.time()
		
		while StopCondt == False:
			iterN += 1
			DualMaster.update()
			DualMaster.optimize()
			#print(DualMaster.status)
			UB = DualMaster.objval
			mu_old = mu_soln.copy()
			if DualMethod == 'cutpl' or (DualMethod == 'level' and iterN == 1):
				for j in range(self.Qlen):
					for i in self.xind[j]:
						mu_soln[j,i] = mu[j,i].x
			elif DualMethod == 'level':
				lt = UB-0.3*(UB-LB)
				levelCon = DualMaster.addConstr(DualObj >= lt)
				DualMaster.setObjective(-quicksum((mu[j,i]-mu_old[j,i])*(mu[j,i]-mu_old[j,i]) for j in range(self.Qlen) for i in self.xind[j]))
				DualMaster.params.Method = 2
				DualMaster.update()
				DualMaster.optimize()
				if DualMaster.status == 2:
					for j in range(self.Qlen):
						for i in self.xind[j]:
							mu_soln[j,i] = mu[j,i].x
				else:
					break
				DualMaster.remove(levelCon)
				DualMaster.setObjective(DualObj)
				DualMaster.params.Method = -1
			if UB-LB < 1e-6*(min(abs(UB),abs(LB))+1):
				StopCondt = True
				print('Dual problem terminates!')

			# Solve "pricing" subproblem
			for j in range(self.Qlen):
				subIP[j].setObjective(quicksum(mu_soln[j,i]*subx[j][i] for i in self.xind[j]))
				subIP[j].update()
				subt0 = time.time()
				subIP[j].optimize()
				subtime += time.time()-subt0
				if subIP[j].status == 2:
					subx_soln = {i:subx[j][i].x for i in self.xind[j]}
					DualMaster.addConstr(theta[j] <= quicksum(mu[j,i]*subx_soln[i] for i in self.xind[j]))
					DWCuts[j] = np.append(DWCuts[j],np.array([[mu_soln[j,i] for i in self.xind[j]]+[-subIP[j].objval]]),axis=0)
					soln = [subx[j][ii].x for ii in self.xind[j]]+[1.0]
					if soln not in SubIPSolns[j].tolist():
						SubIPSolns[j] = np.append(SubIPSolns[j],[soln],axis=0)
				else:
					print('subIP error!')
			LB_new = sum(subIP[j].objval for j in range(self.Qlen))
			if LB_new > LB:
				LB = LB_new
			print(iterN,LB,UB,time.time()-t0,subtime)
		t_Dual = time.time()-t0_Dual

		DualUB = UB
		DualLB = LB
		subIPObj = {j:subIP[j].ObjBound for j in range(self.Qlen)}

		
		# Test up to depth 3 tilting
		DepthUB = 0
		TightAtCurrentNode = {(j,0,0):([{ii:subx[j][ii].x for ii in self.xind[j]}],None) for j in range(self.Qlen)}
		
		# Apply disjunctive coefficient strengthening to DWF cuts
		CTiltCut = {}
		nTilt = 0
		nCSubIPs = 0
		t0_CTilt = time.time()
		for j in range(self.Qlen):
			cut = {}
			for i in self.xind[j]:
				cut[i] = mu_soln[j,i]
			cutRHS = subIP[j].ObjBound


			CoordStatus = {}
			for i in self.xind[j]:
				if subx[j][i].x > 0.5:
					CoordStatus[i] = 1
				else:
					CoordStatus[i] = 0
			

			# CoordStatus[j] = 0(/1) means all known feasible points have its j-th coordinate == 0(/1)
			# CoordStatus[j] = 2 means there are known feasible points having its j-th coordinate equal to 0 and equal to 1, in which case one cannot tilt

			for i in self.xind[j]:
				if CoordStatus[i] == 1:
					subx[j][i].ub = 0.0
					subIP[j].setObjective(quicksum(cut[ii]*subx[j][ii] for ii in self.xind[j]))
					subIP[j].update()
					subIP[j].optimize()
					nCSubIPs += 1
					TightAtCurrentNode[j,0,0][0].append({ii:subx[j][ii].x for ii in self.xind[j]})
					soln = [subx[j][ii].x for ii in self.xind[j]]+[1.0]
					if soln not in SubIPSolns[j].tolist():
						SubIPSolns[j] = np.append(SubIPSolns[j],[soln],axis=0)
					delta = subIP[j].ObjBound - cutRHS
					#print(delta)
					if delta > 1e-5:
						cutRHS += delta
						cut[i] += delta
						nTilt += 1
					for ii in self.xind[j]:
						if CoordStatus[ii] == 0 and subx[j][ii].x > 0.5:
							CoordStatus[ii] = 2
						elif CoordStatus[ii] == 1 and subx[j][ii].x < 0.5:
							CoordStatus[ii] = 2
					subx[j][i].ub = 1.0
					subIP[j].update()
				elif CoordStatus[i] == 0:
					subx[j][i].lb = 1.0
					subIP[j].setObjective(quicksum(cut[ii]*subx[j][ii] for ii in self.xind[j]))
					subIP[j].update()
					subIP[j].optimize()
					nCSubIPs += 1
					TightAtCurrentNode[j,0,0][0].append({ii:subx[j][ii].x for ii in self.xind[j]})
					soln = [subx[j][ii].x for ii in self.xind[j]]+[1.0]
					if soln not in SubIPSolns[j].tolist():
						SubIPSolns[j] = np.append(SubIPSolns[j],[soln],axis=0)
					delta = subIP[j].ObjBound - cutRHS
					if delta > 1e-5:
						cut[i] -= delta
						nTilt += 1
					for ii in self.xind[j]:
						if CoordStatus[ii] == 0 and subx[j][ii].x > 0.5:
							CoordStatus[ii] = 2
						elif CoordStatus[ii] == 1 and subx[j][ii].x < 0.5:
							CoordStatus[ii] = 2
					subx[j][i].lb = 0.0
					subIP[j].update()


			CTiltCut[j] = {ii:cut[ii] for ii in self.xind[j]}
			CTiltCut[j]["RHS"] = cutRHS
		t_CTilt = time.time()-t0_CTilt
		
		TiltTime = []
		
		
		t0 = time.time()
		TiltedCuts = {(j,d):[] for j in range(self.Qlen) for d in range(DepthUB+1)}
		for j in range(self.Qlen):
			TiltedCuts[j,0].append(CTiltCut[j].copy())

			
		
		# Tilt the cuts
		# objval vs ObjBound !!!
		for d in range(DepthUB):
			if d >= 1:
				TiltTime.append(time.time()-t0)
			for j in range(self.Qlen):
				print(d,j,len(TiltedCuts[j,d]))
				for cutInd in range(len(TiltedCuts[j,d])):
					cut = TiltedCuts[j,d][cutInd]
					
					tightInd = (j,d,cutInd)
					tight = TightAtCurrentNode[tightInd][0]
					# backtracking for tight points
					while TightAtCurrentNode[tightInd][1] != None:
						tightInd = TightAtCurrentNode[tightInd][1]
						tight = tight+TightAtCurrentNode[tightInd][0]
					# Determine if there is sufficient degree of freedom for tilting
					rank = np.linalg.matrix_rank(np.array([list(poi.values()) for poi in tight]))
					# print(len(self.xind[j]),rank,len(tight))
					if len(self.xind[j]) <= rank:
						TiltedCuts[j,d+1].append(cut.copy())
						TightAtCurrentNode[j,d+1,len(TiltedCuts[j,d+1])-1] = ([],(j,d,cutInd))
						continue
					SearchLP = Model()
					SearchLP.params.OutputFlag = 0
					SearchLP.params.threads = 4
					v = SearchLP.addVars(self.xind[j],lb=-1,ub=1)
					w = SearchLP.addVar()
					for poi in tight:
						SearchLP.addConstr(quicksum(poi[ii]*v[ii] for ii in self.xind[j]) == w)
					NegVio = SubIPSolns[j].dot([cut[ii] for ii in self.xind[j]]+[-cut["RHS"]])
					FoundCand = False
					for kk in range(len(NegVio)):
						if NegVio[-kk] > 1e-6:
							cand = {self.xind[j][k]:SubIPSolns[j][-kk][k] for k in range(len(self.xind[j]))}
							FoundCand = True
							break
					
					if FoundCand == False:
						subIP[j].setObjective(-quicksum(cut[ii]*subx[j][ii] for ii in self.xind[j]))
						subIP[j].update()
						subIP[j].optimize()
						nSubIPsolved += 1
						cand = {ii:subx[j][ii].x for ii in self.xind[j]}
					

					# Search for tilting direction
					SearchLP.addConstr(quicksum(cand[ii]*v[ii] for ii in self.xind[j]) == w)
					FoundDir = False
					aim = 0
					sign = 1
					
					while FoundDir == False:
						# Optimize over direction e_1, e_2,..., e_n, -e_1, -e_2,..., -e_N
						SearchLP.setObjective(sign*v[self.xind[j][aim]])
						SearchLP.update()
						SearchLP.optimize()


						Dir = [v[ii].x for ii in self.xind[j]]+[w.x]
						if sum(abs(Dir[k]) for k in range(len(self.xind[j]))) <= 1e-8:
							if aim < len(self.xind[j])-1:
								aim += 1
							else:
								aim = 0
								sign = -1
						else:
							FoundDir = True
					
					# a_vec = np.array([cut[ii] for ii in self.xind[j]])
					# v_vec = np.array([v[ii].x for ii in self.xind[j]])
					# print(sum(cut[ii]*v[ii].x for ii in self.xind[j])/np.linalg.norm(a_vec)/np.linalg.norm(v_vec),'!')

					vk1 = {ii:v[ii].x for ii in self.xind[j]}
					wk1 = w.x
					vk2 = {ii:-v[ii].x for ii in self.xind[j]}
					wk2 = -w.x
					FoundValid = False
					lamk1 = float('inf')
					lamk2 = float('inf')
					new_soln1 = {ii:cand[ii] for ii in self.xind[j]}
					new_soln2 = {ii:cand[ii] for ii in self.xind[j]}


					# Use known integer solutions to initialize lamk1 and lamk2
					abVio = SubIPSolns[j].dot([cut[ii] for ii in self.xind[j]]+[-cut["RHS"]])
					vwVio = SubIPSolns[j].dot([v[ii].x for ii in self.xind[j]]+[-w.x])
					for kk in range(len(SubIPSolns[j])):
						if vwVio[kk] < -1e-8 and lamk1>-abVio[kk]/vwVio[kk]:
							lamk1 = -abVio[kk]/vwVio[kk]
							new_soln1 = {self.xind[j][k]:SubIPSolns[j][kk][k] for k in range(len(self.xind[j]))}
						elif vwVio[kk] > 1e-8 and lamk2>abVio[kk]/vwVio[kk]:
							lamk2 = abVio[kk]/vwVio[kk]
							new_soln2 = {self.xind[j][k]:SubIPSolns[j][kk][k] for k in range(len(self.xind[j]))}
					if lamk1 < float('inf') and lamk1 >= 1e-8:
						vk1 = {ii:cut[ii]+lamk1*v[ii].x for ii in self.xind[j]}
						wk1 = cut["RHS"]+lamk1*w.x
					elif lamk1 < 1e-8:
						vk1 = {ii:cut[ii] for ii in self.xind[j]}
						wk1 = cut["RHS"]
					if lamk2 < float('inf') and lamk2 >= 1e-8:
						vk2 = {ii:cut[ii]-lamk2*v[ii].x for ii in self.xind[j]}
						wk2 = cut["RHS"]-lamk2*w.x
					elif lamk2 < 1e-8:
						vk2 = {ii:cut[ii] for ii in self.xind[j]}
						wk2 = cut["RHS"]
					
					lamk1old = lamk1
					lamk2old = lamk2


					if lamk1 >= 1e-8:
						# Diretion v^Tx >= w
						while FoundValid == False:
							print(tightInd,'lamk1',lamk1)
							# print(vk1,wk1)
							if lamk1 < 1e-8 or (lamk1 > lamk1old and lamk1old < 1e-4):
								FoundValid = True
								vk1 = {ii:cut[ii] for ii in self.xind[j]}
								wk1 = cut["RHS"]
								new_soln1 = {ii:subx[j][ii].x for ii in self.xind[j]}
							else:
								subIP[j].setObjective(quicksum(vk1[ii]*subx[j][ii] for ii in self.xind[j]))
								subIP[j].update()
								subIP[j].optimize()
								nSubIPsolved += 1
								soln = [subx[j][ii].x for ii in self.xind[j]]+[1.0]
								if soln not in SubIPSolns[j].tolist():
									SubIPSolns[j]=np.append(SubIPSolns[j],[soln],axis=0)
								if subIP[j].objval < wk1-1e-5:
									# print(sum(abs(vk1[ii]) for ii in self.xind[j]))
									# print(subIP[j].objval,wk1)
									# vk= a+lamk1*v, wk = b+lamk1*w, lamk1 = (a^Tx^*-b)/(w-v^Tx^*)
									if abs(sum(v[ii].x*subx[j][ii].x for ii in self.xind[j])-w.x) < 1e-5 and subIP[j].objval >= wk1-1e-3:
										#print(i,k,d,cutInd,len(TiltedCuts[i,k,d+1]))
										wk1 = subIP[j].ObjBound
										FoundValid = True
										break
									lamk1old = lamk1
									lamk1 = (sum(cut[ii]*subx[j][ii].x for ii in self.xind[j])-cut["RHS"])/(w.x-sum(v[ii].x*subx[j][ii].x for ii in self.xind[j]))
									# print(lamk1,(sum(cut[ii]*subx[j][ii].x for ii in self.xind[j])-cut["RHS"]),(w.x-sum(v[ii].x*subx[j][ii].x for ii in self.xind[j])))
									vk1 = {ii:cut[ii]+lamk1*v[ii].x for ii in self.xind[j]}
									wk1 = cut["RHS"]+lamk1*w.x
									new_soln1 = {ii:subx[j][ii].x for ii in self.xind[j]}
								else:
									wk1 = subIP[j].ObjBound
									FoundValid = True
					
					if lamk2 >= 1e-8:
						# Diretion -v^Tx >= -w
						FoundValid = False
						while FoundValid == False:
							print(tightInd,'lamk2',lamk2)
							if lamk2 < 1e-8 or (lamk2 > lamk2old and lamk2old < 1e-4):
								FoundValid = True
								vk2 = {ii:cut[ii] for ii in self.xind[j]}
								wk2 = cut["RHS"]
								new_soln2 = {ii:subx[j][ii].x for ii in self.xind[j]}
							else:
								subIP[j].setObjective(quicksum(vk2[ii]*subx[j][ii] for ii in self.xind[j]))
								subIP[j].update()
								subIP[j].optimize()
								nSubIPsolved += 1
								soln = [subx[j][ii].x for ii in self.xind[j]]+[1.0]
								if soln not in SubIPSolns[j].tolist():
									SubIPSolns[j]=np.append(SubIPSolns[j],[soln],axis=0)
								if subIP[j].objval < wk2-1e-5:
									# vk= a-lamk2*v, wk = b-lamk2*w, lamk2 = (a^Tx^*-b)/(v^Tx^*-w)
									# print(subIP[j].objval,wk2)
									if abs(sum(v[ii].x*subx[j][ii].x for ii in self.xind[j])-w.x) < 1e-5 and subIP[j].objval >= wk2-1e-3:
										#print(i,k,d,cutInd,len(TiltedCuts[i,k,d+1]))
										wk2 = subIP[j].ObjBound
										FoundValid = True
										break
									lamk2old = lamk2
									lamk2 = (sum(cut[ii]*subx[j][ii].x for ii in self.xind[j])-cut["RHS"])/(sum(v[ii].x*subx[j][ii].x for ii in self.xind[j])-w.x)
									# print(lamk2,(sum(cut[ii]*subx[j][ii].x for ii in self.xind[j])-cut["RHS"]),(w.x-sum(v[ii].x*subx[j][ii].x for ii in self.xind[j])))
									vk2 = {ii:cut[ii]-lamk2*v[ii].x for ii in self.xind[j]}
									wk2 = cut["RHS"]-lamk2*w.x
									new_soln2 = {ii:subx[j][ii].x for ii in self.xind[j]}
								else:
									wk2 = subIP[j].ObjBound
									FoundValid = True

					cut = vk1
					cut["RHS"] = wk1
					TiltedCuts[j,d+1].append(cut.copy())
					TightAtCurrentNode[j,d+1,len(TiltedCuts[j,d+1])-1] = ([new_soln1],(j,d,cutInd))
					#subIP[i,k].setObjective(quicksum(vk1[ii]*subx[i,k][Sk[k][ii]] for ii in range(int(self.n/self.r)))+vk1[int(self.n/self.r)]*suby[i,k])
					#subIP[i,k].update()
					#subIP[i,k].optimize()
					#if subIP[i,k].objval < wk1-1e-8:
					#	print(d,i,k,lamk1,len(TiltedCuts[i,k,d+1])-1,subIP[i,k].objval,wk1,'*')

					if lamk1 >= 1e-8 or lamk2 >= 1e-8:
						cut = vk2
						cut["RHS"] = wk2
						TiltedCuts[j,d+1].append(cut.copy())
						TightAtCurrentNode[j,d+1,len(TiltedCuts[j,d+1])-1] = ([new_soln2],(j,d,cutInd))

		TiltTime.append(time.time()-t0)

		

		# Calculate root bound
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.n,vtype=GRB.BINARY)
		m.setObjective(-quicksum(self.p[i]*x[i] for i in range(self.n)))
		m.addConstrs((quicksum(self.w[i]*x[i] for i in self.S[j])<= self.C for j in range(self.n)))
		m.update()
		m_LP = m.relax()
		m_LP.update()
		m_LP.optimize()
		LB_LP = m_LP.ObjBound
		m.params.NodeLimit = 1
		m.update()
		t0 = time.time()
		m.optimize()
		Root0 = m.ObjBound
		RootTime0 = time.time()-t0
		del m_LP
		del m
		
		# Original formulation
		m = Model()
		m.params.threads = 4
		x = m.addVars(self.n,vtype=GRB.BINARY)
		m.setObjective(-quicksum(self.p[i]*x[i] for i in range(self.n)))
		m.addConstrs((quicksum(self.w[i]*x[i] for i in self.S[j]) <= self.C for j in range(self.n)))
		m.params.TimeLimit = tLimit
		m.params.LogFile = "LogFiles/TKP/TKP_"+self.instanceName+"_"+str(self.Qsize)+"_"+str(tLimit)+"_original.log"
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

		# Solve OBJ root
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.n,vtype=GRB.BINARY)
		m.setObjective(-quicksum(self.p[i]*x[i] for i in range(self.n)))
		m.addConstrs((quicksum(self.w[i]*x[i] for i in self.S[j])<= self.C for j in range(self.n)))
		m.addConstr(-quicksum(self.p[i]*x[i] for i in range(self.n)) >= LB)
		m.params.NodeLimit = 1
		m.update()
		t0 = time.time()
		m.optimize()
		Root1 = m.ObjBound
		RootTime1 = time.time()-t0
		del m

		# Solve OBJ model
		m = Model()
		m.params.threads = 4
		x = m.addVars(self.n,vtype=GRB.BINARY)
		m.setObjective(-quicksum(self.p[i]*x[i] for i in range(self.n)))
		m.addConstrs((quicksum(self.w[i]*x[i] for i in self.S[j])<= self.C for j in range(self.n)))
		m.addConstr(-quicksum(self.p[i]*x[i] for i in range(self.n)) >= LB)
		m.params.TimeLimit = tLimit
		m.params.LogFile = "LogFiles/TKP/TKP_"+self.instanceName+"_"+str(self.Qsize)+"_"+str(tLimit)+"_objLB.log"
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

		# Solve DWF root (DWF cuts as regular cuts)
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.n,vtype=GRB.BINARY)
		m.setObjective(-quicksum(self.p[i]*x[i] for i in range(self.n)))
		m.addConstrs((quicksum(self.w[i]*x[i] for i in self.S[j])<= self.C for j in range(self.n)))
		for j in range(self.Qlen):
			m.addConstr(quicksum(mu_soln[j,i]*x[i] for i in self.xind[j]) >= subIPObj[j])
		m.params.NodeLimit = 1
		m.update()
		t0 = time.time()
		m.optimize()
		Root2 = m.ObjBound
		RootTime2 = time.time()-t0
		del m

		# Solve DWF model (DWF cuts as regular cuts)
		m = Model()
		m.params.threads = 4
		x = m.addVars(self.n,vtype=GRB.BINARY)
		m.setObjective(-quicksum(self.p[i]*x[i] for i in range(self.n)))
		m.addConstrs((quicksum(self.w[i]*x[i] for i in self.S[j])<= self.C for j in range(self.n)))
		for j in range(self.Qlen):
			m.addConstr(quicksum(mu_soln[j,i]*x[i] for i in self.xind[j]) >= subIPObj[j])
		m.params.TimeLimit = tLimit
		m.params.LogFile = "LogFiles/TKP/TKP_"+self.instanceName+"_"+str(self.Qsize)+"_"+str(tLimit)+"_DWcuts.log"
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

		# Solve STR root
		m = Model()
		m.params.OutputFlag = 0
		m.params.threads = 4
		x = m.addVars(self.n,vtype=GRB.BINARY)
		m.setObjective(-quicksum(self.p[i]*x[i] for i in range(self.n)))
		m.addConstrs((quicksum(self.w[i]*x[i] for i in self.S[j])<= self.C for j in range(self.n)))
		for j in range(self.Qlen):
			m.addConstr(quicksum(CTiltCut[j][i]*x[i] for i in self.xind[j]) >= CTiltCut[j]["RHS"])
		m.params.NodeLimit = 1
		m.update()
		t0 = time.time()
		m.optimize()
		Root3 = m.ObjBound
		RootTime3 = time.time()-t0
		del m

		# Solve STR model
		m = Model()
		m.params.threads = 4
		x = m.addVars(self.n,vtype=GRB.BINARY)
		m.setObjective(-quicksum(self.p[i]*x[i] for i in range(self.n)))
		m.addConstrs((quicksum(self.w[i]*x[i] for i in self.S[j])<= self.C for j in range(self.n)))
		for j in range(self.Qlen):
			m.addConstr(quicksum(CTiltCut[j][i]*x[i] for i in self.xind[j]) >= CTiltCut[j]["RHS"])
		m.params.TimeLimit = tLimit
		m.params.LogFile = "LogFiles/TKP/TKP_"+self.instanceName+"_"+str(self.Qsize)+"_"+str(tLimit)+"_DWcutsCTp.log"
		m.update()
		t0IP3 = time.time()
		m.optimize()
		t_IP3 = time.time()-t0IP3
		if m.status == 2:
			ifSolve3 = 1
		else:
			ifSolve3 = 0
		nNodes3 = m.NodeCount
		Gap3 = m.MIPGap
		LB3 = m.ObjBound
		UB3 = m.objval
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
			x = m.addVars(self.n,vtype=GRB.BINARY)
			m.setObjective(-quicksum(self.p[i]*x[i] for i in range(self.n)))
			m.addConstrs((quicksum(self.w[i]*x[i] for i in self.S[j])<= self.C for j in range(self.n)))
			for j in range(self.Qlen):
				for cut in TiltedCuts[j,d]:
					m.addConstr(quicksum(cut[ii]*x[ii] for ii in self.xind[j]) >= cut["RHS"])
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
			x = m.addVars(self.n,vtype=GRB.BINARY)
			m.setObjective(-quicksum(self.p[i]*x[i] for i in range(self.n)))
			m.addConstrs((quicksum(self.w[i]*x[i] for i in self.S[j])<= self.C for j in range(self.n)))
			for j in range(self.Qlen):
				for cut in TiltedCuts[j,d]:
					m.addConstr(quicksum(cut[ii]*x[ii] for ii in self.xind[j]) >= cut["RHS"])
			m.params.TimeLimit = tLimit
			m.params.LogFile = "LogFiles/TKP/TKP_"+self.instanceName+"_"+str(self.Qsize)+"_DWcutsCTp"+str(d)+"_"+str(tLimit)+".log"
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

		Output = [LB_LP,LB,DualLB,DualUB,t_Dual,iterN,Root0,Root1,Root2,Root3]+RootCTd+[RootTime0,RootTime1,RootTime2,RootTime3]+RootTimeCTd+[ifSolve0,ifSolve1,ifSolve2,ifSolve3]+ifSolveCTd+[t_IP0,t_IP1,t_IP2,t_IP3]+t_IPCTd+\
			[nNodes0,nNodes1,nNodes2,nNodes3]+nNodesCTd+[LB0,LB1,LB2,LB3]+LBCTd+[UB0,UB1,UB2,UB3]+UBCTd+[Gap0,Gap1,Gap2,Gap3]+GapCTd+[t_CTilt]+TiltTime+[nTilt,nCSubIPs]
		OutputStr = ""
		for i in range(len(Output)):
			OutputStr = OutputStr+str(Output[i])+'\t'
		return OutputStr

for InstanceClass in ['U']:#['I','U']:
	for ind in range(80,101):#range(1,101):
		instance = InstanceClass+str(ind)
		for Qsize in [32,64]:
			tLimit = 10*60
			ss = TKP(instance,Qsize)
			ss.readInstance()
			OutputStr = ss.ExtCompare(tLimit=tLimit)
			WrtStr = instance+'\t'+OutputStr+'\n'
			f = open('TKP_ExtCompare'+str(Qsize)+'_'+str(tLimit)+'.txt','a')
			f.write(WrtStr)
			f.close()