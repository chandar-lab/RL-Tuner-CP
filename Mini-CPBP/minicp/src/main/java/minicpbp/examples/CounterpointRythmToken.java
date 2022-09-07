package minicpbp.examples;

import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.util.exception.InconsistencyException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static minicpbp.cp.Factory.*;

public class CounterpointRythmToken {
    private static Solver cp;
    private static int n = 14;
    private static boolean useRandomSequence = false;
    private static int maxDuration = 4;
    // Rythms allowed: sixteenth-note, eight-note, quarter-note, half-note, full-note
    private static int[] rythmDomain = {1, 2, 4};

    public static void main(String[] args) {
        int sumDurations = args.length;
        ArrayList<Integer> previousDurations = new ArrayList<>();
        int index = 0;
        while(index < args.length) {
            if (Integer.parseInt(args[index]) > 0) {
                int note = Integer.parseInt(args[index]);
                int duration = 0;
                while (index < args.length && note == Integer.parseInt(args[index])) {
                    duration++;
                    index++;
                }
                previousDurations.add(duration);
            } else
                index++;
        }


        if (useRandomSequence) {
            Random random = new Random();
            int nNotesInArgs = random.nextInt(14) + 1;
            previousDurations = new ArrayList<>();
            System.out.print("Args: ");
            for (int i = 0; i < nNotesInArgs; i++) {
                int note = random.nextInt(28) + 1;
                int duration = random.nextInt(maxDuration) + 1;
                previousDurations.add(duration);
                for (int j = 0; j < duration; j++) {
                    System.out.print(note + " ");
                }
            }
            System.out.println();

            sumDurations = previousDurations.stream().mapToInt(a -> a).sum();
        }

        int currentDuration = 0;
        if(previousDurations.size() > 0) {
            currentDuration = previousDurations.get(previousDurations.size() - 1);
            previousDurations.remove(previousDurations.size() - 1);
        }
        sumDurations -= currentDuration;

        double[] marginals = new double[maxDuration];

        cp = makeSolver();
        ArrayList<Integer> finalPreviousDurations = previousDurations;
        IntVar[] durationsFuture = makeIntVarArray(n - previousDurations.size(),
                (i) -> makeIntVar(cp, Arrays.stream(rythmDomain).boxed().collect(Collectors.toSet())));

        try {
            // Rythm constraints
            //rythm(previousDurations, durationsFuture);
            maxTokens(sumDurations, durationsFuture);
            smallDurations(previousDurations, durationsFuture);
            initializeMarginals(durationsFuture, currentDuration);
            //printTime();
            solve();
            //printTime()
            IntVar currentNoteMarginals = durationsFuture[0];

            int[] domain = new int[currentNoteMarginals.size()];
            currentNoteMarginals.fillArray(domain);

            for(int value : domain)
                marginals[value - 1] = currentNoteMarginals.marginal(value);

            printMarginals(marginals);
            //main(null);
            // If there is a contradiction in the constraints, it means that there is no solution
        } catch(InconsistencyException e) {
            printMarginals(marginals);
            //main(null);
        }
    }

    private static void smallDurations(ArrayList<Integer> previousDurations, IntVar[] durationsFuture) {
        int durationOne = 0;
        int durationTwo = 0;
        int durationFour = 0;

        for(int i = 0; i < previousDurations.size(); i++) {
            if (previousDurations.get(i) == 1)
                durationOne++;
            else if (previousDurations.get(i) == 2)
                durationTwo++;
            else if (previousDurations.get(i) == 4)
                durationFour++;
        }
        IntVar[] occurrences = makeIntVarArray(cp, 3, 0, 56);
        cp.post(lessOrEqual(sum(occurrences[2], makeIntVar(cp, durationFour, durationFour)), sum(occurrences[1], makeIntVar(cp, durationTwo, durationTwo))));
        cp.post(lessOrEqual(sum(occurrences[1], makeIntVar(cp, durationTwo, durationTwo)), sum(occurrences[0], makeIntVar(cp, durationOne, durationOne))));
        cp.post(cardinality(durationsFuture, new int[]{1, 2, 4}, occurrences));

    }

    private static void maxTokens(int sumDurations, IntVar[] durationsFuture) {
        int remainingTokens = (n * maxDuration) - sumDurations;
        cp.post(lessOrEqual(sum(durationsFuture), makeIntVar(cp, remainingTokens, remainingTokens)));
    }

    private static void rythm(ArrayList<Integer> previousDurations, IntVar[] durationsFuture) {
        int maxOfShortNotes = 0;
        int minOfLongNotes = Integer.MAX_VALUE;
        for (int i = 0; i < previousDurations.size(); i++) {
            if (i % 3 == 2) {
                if (previousDurations.get(i) < minOfLongNotes) {
                    minOfLongNotes = previousDurations.get(i);
                }
            } else {
                if (previousDurations.get(i) > maxOfShortNotes) {
                    maxOfShortNotes = previousDurations.get(i);
                }
            }
        }
        if(maxOfShortNotes == maxDuration)
            maxOfShortNotes--;

        if(minOfLongNotes == 1)
            minOfLongNotes++;

        boolean[] longNotes = new boolean[]{
                false, false, true, false, false, true, false, false, true, false, false, true, false, false};
        for (int i = 0; i < durationsFuture.length; i++) {
            boolean isLongI = longNotes[previousDurations.size() + i];

            if(isLongI) {
                cp.post(larger(durationsFuture[i], makeIntVar(cp, maxOfShortNotes, maxOfShortNotes)));
            } else {
                cp.post(less(durationsFuture[i], makeIntVar(cp, minOfLongNotes, minOfLongNotes)));
            }

            if (i != durationsFuture.length - 1) {
                for (int j = i + 1; j < durationsFuture.length; j++) {
                    boolean isLongJ = longNotes[previousDurations.size() + j];
                    if (isLongI && !isLongJ) {
                        cp.post(larger(durationsFuture[i], durationsFuture[j]));
                    }
                    if (!isLongI && isLongJ) {
                        cp.post(larger(durationsFuture[j], durationsFuture[i]));
                    }
                }
            }
        }
    }

    private static void solve() {
        cp.fixPoint();
        cp.beliefPropa();
    }

    private static void initializeMarginals(IntVar[] durations, int currentDuration) {
        durations[0].removeBelow(currentDuration);
    }

    public static void printMarginals(double[] marginals) {
        int i = 0;
        for (double m : marginals) {
            System.out.println(m);
            i++;
        }
    }

}
