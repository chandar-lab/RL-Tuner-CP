package fzn;

import minicpbp.engine.core.Constraint;
import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.engine.core.Solver.PropaMode;
import minicpbp.search.Search;
import minicpbp.search.SearchStatistics;
import minicpbp.util.Procedure;
import minicpbp.util.exception.InconsistencyException;
import minicpbp.engine.core.MiniCP;

import launch.SolveXCSPFZN.BranchingHeuristic;
import launch.SolveXCSPFZN.TreeSearchType;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.*;
import java.util.function.Supplier;

import fzn.parser.FZParser;
import fzn.parser.intermediatemodel.*;

import static minicpbp.cp.BranchingScheme.and;
import static minicpbp.cp.BranchingScheme.firstFail;
import static minicpbp.cp.BranchingScheme.firstFailRandomVal;
import static minicpbp.cp.BranchingScheme.maxMarginalStrength;
import static minicpbp.cp.BranchingScheme.maxMarginal;
import static minicpbp.cp.BranchingScheme.minMarginalStrength;
import static minicpbp.cp.BranchingScheme.minMarginal;
import static minicpbp.cp.BranchingScheme.minEntropy;
import static minicpbp.cp.BranchingScheme.impactEntropy;
import static minicpbp.cp.BranchingScheme.minEntropyRegisterImpact;
import static minicpbp.cp.BranchingScheme.minEntropyBiasedWheelSelectVal;
import static minicpbp.cp.Factory.*;
import static java.lang.reflect.Array.newInstance;

public class FZN {

    private String fileName;
	private final List<IntVar> minicpVars = new LinkedList<>();


	private final Solver minicp = makeSolver();

	private Optional<IntVar> objectiveMinimize = Optional.empty();
	private Optional<IntVar> realObjective = Optional.empty();

	//Model containing all constraints, parameters, variables and functions of the problem
	private Model m;

	private boolean hasFailed;

    public FZN(String filename) throws Exception {
        this.fileName = filename;
        hasFailed = false;
		//read the Flatzinc File
		this.m = FZParser.readFlatZincModelFromFile(filename, false);
    }

	/**
	 * 
	 * @return true if the problem is a COP, false if the problem is a CSP
	 */
    public boolean isCOP() {
		return objectiveMinimize.isPresent();
	}

	private String solutionStr = null;
	private boolean extractSolutionStr = false;
	private boolean foundSolution = false;

	private static boolean checkSolution = false;

	public void checkSolution(boolean checkSolution) {
		FZN.checkSolution = checkSolution;
	}

	private static boolean traceBP = false;

	public void traceBP(boolean traceBP) {
		FZN.traceBP = traceBP;
	}

	private static boolean traceSearch = false;

	public void traceSearch(boolean traceSearch) {
		FZN.traceSearch = traceSearch;
	}

	private static int maxIter = 5;

	public void maxIter(int maxIter) {
		FZN.maxIter = maxIter;
	}

	private static boolean damp = false;

	public void damp(boolean damp) {
		FZN.damp = damp;
	}

	private static double dampingFactor = 0.5;

	public void dampingFactor(double dampingFactor) {
		FZN.dampingFactor = dampingFactor;
	}

	private static TreeSearchType searchType = TreeSearchType.DFS;

	public void searchType(TreeSearchType searchType) {
		FZN.searchType = searchType;
	}

	private static boolean restart = false;

	public void restart(boolean restart) {
		FZN.restart = restart;
	}

	private static int nbFailCutof = 100;
	
	public void nbFailCutof(int nbFailCutof) {
		FZN.nbFailCutof = nbFailCutof;
	} 

	private static double restartFactor = 1.5;

	public void restartFactor(double restartFactor) {
		FZN.restartFactor = restartFactor;
	}

	private static double variationThreshold = -Double.MAX_VALUE;

	public void variationThreshold(double variationThreshold) {
		FZN.variationThreshold = variationThreshold;
	}

	private static boolean initImpact = false;

	public void initImpact(boolean initImpact) {
		FZN.initImpact = initImpact;
	}

	private static boolean dynamicStopBP = false;

	public void dynamicStopBP(boolean dynamicStopBP) {
		FZN.dynamicStopBP = dynamicStopBP;
	}

	private static boolean traceNbIter = false;

	public void traceNbIter(boolean traceNbIter) {
		FZN.traceNbIter = traceNbIter;
	}

	/**
	 * Creates a search (either DFS or LDS) with a given branching heuristic
	 * @param branching a branching heuristic
	 * @return a search object
	 */
	private Search makeSearch(Supplier<Procedure[]> branching) {
		Search search = null;
		switch (searchType) {
		case DFS:
			search = makeDfs(minicp, branching);
			break;
		case LDS:
			search = makeLds(minicp, branching);
			break;
		default:
			System.out.println("unknown search type");
			System.exit(1);
		}
		return search;
	}


	public void solve(BranchingHeuristic heuristic, int timeout, String statsFileStr, String solFileStr) {

		Long t0 = System.currentTimeMillis();

		minicp.setTraceBPFlag(traceBP);
		minicp.setTraceSearchFlag(traceSearch);
		minicp.setDynamicStopBP(dynamicStopBP);
		minicp.setTraceNbIterFlag(traceNbIter);
		minicp.setMaxIter(maxIter);
		minicp.setDamp(damp);
		minicp.setDampingFactor(dampingFactor);
		minicp.setVariationThreshold(variationThreshold);

		if (hasFailed) {
			System.out.println("problem failed before initiating the search");
			throw InconsistencyException.INCONSISTENCY;
		}
	
		m.addSolver(minicp);

		//build the model from the Flatzinc file
		m.buildModel();
		
		Search search = null;
		MiniCP minicpbp = (MiniCP) minicp;
		//create search and branching heuristic
		switch (heuristic) {
		case FFRV:
			minicp.setMode(PropaMode.SP);
			search = makeSearch(firstFailRandomVal(m.getDecisionsVar()));
			break;
		case MXMS:
			search = makeSearch(maxMarginalStrength(m.getDecisionsVar()));
			break;
		case MXM:
			search = makeSearch(maxMarginal(m.getDecisionsVar()));
			break;
		case MNMS:
			search = makeSearch(minMarginalStrength(m.getDecisionsVar()));
			break;
		case MNM:
			search = makeSearch(minMarginal(m.getDecisionsVar()));
			break;
		case MNE:
			search = makeSearch(minEntropy(m.getDecisionsVar()));
			break;
		case IE:
			search = makeSearch(impactEntropy(m.getDecisionsVar()));
			//optional initialisation of impacts
			if(FZN.initImpact)
				search.initializeImpact(m.getDecisionsVar());
			break;
		case MIE:
			search = makeDfs(minicp, minEntropyRegisterImpact(m.getDecisionsVar()),impactEntropy(m.getDecisionsVar()));
			//optional initialisation of impacts
			if(FZN.initImpact)
				search.initializeImpact(m.getDecisionsVar());
			break;
		case MNEBW:
			search = makeSearch(minEntropyBiasedWheelSelectVal(m.getDecisionsVar()));
			break;
		default:
			System.out.println("unknown search strategy");
			System.exit(1);
		}

		if (checkSolution || (solFileStr != ""))
			extractSolutionStr = true;

		//procedure executed when a solution is found
		search.onSolution(() -> {
			foundSolution = true;
			if (extractSolutionStr) {
				StringBuilder sol = new StringBuilder("<instantiation>\n\t<list>\n\t\t");
				for (IntVar x : m.getDecisionsVar())
					sol.append(x.getName()).append(" ");
				sol.append("\n\t</list>\n\t<values>\n\t\t");
				for (IntVar x : m.getDecisionsVar()){
					System.out.println(x.size());
					sol.append(x.min()).append(" ");
				}
				sol.append("\n\t</values>\n</instantiation>");
				solutionStr = sol.toString();
			}
		});
		SearchStatistics stats;
		//start the search
		switch (m.getGoal()) {
			//find a solution that maximizes the cost function
			case ASTSolve.MAX:
				stats = search.optimize(minicpbp.maximize(m.getObjective()),
					ss -> {
						return (System.currentTimeMillis() - t0 >= timeout * 1000 || foundSolution);
					});
				break;
			//find a solution that minimizes the cost function
			case ASTSolve.MIN:
				stats = search.optimize(minicpbp.minimize(m.getObjective()),
				ss -> {
					return (System.currentTimeMillis() - t0 >= timeout * 1000 || foundSolution);
				});
				break;
			default:
				//find a solution that satisfies all constraints without restart
				if(!restart) {
					stats = search.solve(ss -> {
						return (System.currentTimeMillis() - t0 >= timeout * 1000 || foundSolution);
					});
				}
				//find a solution that satisfies all constraints with restarts during the search
				else {
					stats = search.solveRestarts(ss -> {
						return (System.currentTimeMillis() - t0 >= timeout * 1000 || foundSolution);
					}, nbFailCutof, restartFactor);
				}
				break;
		}

		if (foundSolution) {
			System.out.println("solution found");
			printSolution(solFileStr);
		} else
			System.out.println("no solution was found");

		Long runtime = System.currentTimeMillis() - t0;
		printStats(stats, statsFileStr, runtime);

	}

	/**
	 * Prints the solution in the given file
	 * @param solFileStr the path to the file
	 */
	private void printSolution(String solFileStr) {
		if (solFileStr != "")
			try {
				PrintWriter out = new PrintWriter(new File(solFileStr));
				out.print(solutionStr);
				out.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				System.out.println("unable to create file " + solFileStr);
				System.exit(1);
			}
	}

	/**
	 * Prints statistics about the search
	 * @param stats statistics about the search
	 * @param statsFileStr a path to save the stats 
	 * @param runtime
	 */
	private void printStats(SearchStatistics stats, String statsFileStr, Long runtime) {
		PrintStream out = null;
		if (statsFileStr == "")
			out = System.out;
		else
			try {
				out = new PrintStream(new File(statsFileStr));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				System.out.println("unable to create file " + statsFileStr);
				System.exit(1);
			}

		String statusStr;
		if (foundSolution)
			statusStr = "SAT";
		else if (stats.isCompleted())
			statusStr = "UNSAT";
		else
			statusStr = "TIMEOUT";

		out.println("status: " + statusStr);
		out.println("failures: " + stats.numberOfFailures());
		out.println("nodes: " + stats.numberOfNodes());
		out.println("runtime (ms): " + runtime);

		out.close();

	}

}
