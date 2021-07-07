/*
 * mini-cp is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * mini-cp is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c)  2017. by Laurent Michel, Pierre Schaus, Pascal Van Hentenryck
 *
 * mini-cpbp, replacing classic propagation by belief propagation 
 * Copyright (c)  2019. by Gilles Pesant
 */

package minicp.examples;


import minicp.engine.core.IntVar;
import minicp.engine.core.Solver;
import minicp.search.DFSearch;
import minicp.search.SearchStatistics;
import static minicp.cp.Factory.*;
import static minicp.cp.BranchingScheme.*;

import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

/**
 * The Magic Square Completion problem.
 * <a href="http://csplib.org/Problems/prob019/">CSPLib</a>.
 */
public class MagicSquare {

    public static void main(String[] args) {
		// Parameters for the example file
		int n = Integer.parseInt(args[0]);
		int nbFilled = Integer.parseInt(args[1]);
		int nbFile = Integer.parseInt(args[2]);

		boolean notEqual = false; // If decomposing the all different into binary constraints

		// Initializing CP solver
		Solver cp = makeSolver();

		// 1D array representing the magic square to fill
		// This array contains probabilities of all possible values for all cells of the square
		// If a value for a given cell was already specified, the array will contain only one value for that cell
		IntVar[] xFlat = makeMagicSquare(cp,n,notEqual,nbFilled,nbFile);

		// Converting flat array into a 2D square
		IntVar[][] x = new IntVar[n][n];
		for(int i = 0; i<n; i++){
	    	for(int j = 0; j<n; j++){
				x[i][j] = xFlat[i*n+j];
	    	}
		}

//    	DFSearch dfs = makeDfs(cp, firstFailRandomVal(xFlat));
		// Search for solutions using the marginal strength
		// The strength is determined by the value of the marginal compared to a uniformly distributed value
		// Ex: if the marginal is 0.5 and the domain contains only two values, the marginal is weak
		// Ex: if the marginal is 0.5 and the domain contains 5 values, the marginal is strong
   		DFSearch dfs = makeDfs(cp, maxMarginalStrength(xFlat));

		// When a solution is found, an event is sent and the square will be displayed
        dfs.onSolution(() -> {
                    for (int i = 0; i < n; i++) {
                        System.out.println(Arrays.toString(x[i]));
                    }
                }
        );

        // Yes...
        SearchStatistics stats = dfs.solve(stat -> stat.numberOfSolutions() >= 1); // stop on first solution

        System.out.println(stats);

    }

	public static void partialAssignments(IntVar[][] vars, int n, int nbFilled, int nbFile){
	    try {
	    	// Opening the example file
			Scanner scanner = new Scanner(new FileReader("src/main/java/minicp/examples/data/MagicSquare/magicSquare"+n+"-filled"+nbFilled+"-"+nbFile+".dat"));
		
			scanner.nextInt();
			scanner.nextInt();
		
			while(scanner.hasNextInt()){
		    	int row = scanner.nextInt()-1;
		    	int column = scanner.nextInt()-1;
		    	int value = scanner.nextInt();
		    	vars[row][column].assign(value); // Assigning the value of all the cells provided in the file
			}
			scanner.close();
	    }
	    catch (IOException e) {
			System.err.println("Error : " + e.getMessage()) ;
			System.exit(2) ;
	    }
	}

	public static IntVar[] makeMagicSquare(Solver cp, int n, boolean notEqual, int nbFilled, int nbFile) {
		// Values go from 1 to n*n
		// For a given number x, the sum of the numbers from 1 to x is equal to x(x+1)/2
		// The sum of the whole square is n*n(n*n + 1)/2
		// The sum of one line/column/diagonal is the total sum divided by the number of lines
		// (n*n(n*n + 1)/2)/n = n(n*n+1)/2
		int M = n*(n*n+1)/2;

		// Creating empty magic square
		// Every cell is an intvar that contains all possible values from 1 to n^2
		IntVar[][] x = new IntVar[n][n];

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				x[i][j] = makeIntVar(cp, 1, n*n);
				x[i][j].setName("x"+"["+(i+1)+","+(j+1)+"]");
			}
		}

		// Creating a 1D version of the 2D square
		IntVar[] xFlat = new IntVar[x.length * x.length];
		for (int i = 0; i < x.length; i++) {
			System.arraycopy(x[i],0,xFlat,i * x.length,x.length);
		}

		// Filling square with provided values
		partialAssignments(x,n,nbFilled,nbFile);

		// Sum on lines
		for (int i = 0; i < n; i++) {
			cp.post(sum(x[i],M));
		}

		// Sum on columns
		for (int j = 0; j < x.length; j++) {
			IntVar[] column = new IntVar[n];
			for (int i = 0; i < x.length; i++)
				column[i] = x[i][j];
			cp.post(sum(column,M));
		}

		// Sum on diagonals
		IntVar[] diagonalLeft = new IntVar[n];
		IntVar[] diagonalRight = new IntVar[n];
		for (int i = 0; i < x.length; i++){
			diagonalLeft[i] = x[i][i];
			diagonalRight[i] = x[n-i-1][i];
		}
		cp.post(sum(diagonalLeft, M));
		cp.post(sum(diagonalRight, M));

		// AllDifferent
		if(notEqual)
 		    cp.post(allDifferent(xFlat)); // Decomposes the all different constraint into binary constraints
		else
		    cp.post(allDifferentAC(xFlat));

		return xFlat;
	}

}
