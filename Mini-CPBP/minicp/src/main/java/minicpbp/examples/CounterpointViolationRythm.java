package minicpbp.examples;

import minicpbp.engine.core.BoolVar;
import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.util.exception.InconsistencyException;

import java.util.ArrayList;
import java.util.Random;

import static minicpbp.cp.Factory.*;

public class CounterpointViolationRythm {
    private static Solver cp;
    private static int n = 14;
    private static boolean useRandomSequence = false;
    private static int maxDuration = 16;

    public static void main(String[] args) {
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
        }
        /*
        Si on décide de changer de note, il faut prendre en compte la durée finale de la dernière note: ça compte comme une violation potentielle
        Si on décide de prolonger la note, il faut prendre en compte la durée de cette note
         */
        int currentDuration = previousDurations.get(previousDurations.size() - 1);
        previousDurations.remove(previousDurations.size() - 1);
        int pastDuration = 0;
        if (previousDurations.size() > 1 && currentDuration == 1) {
            pastDuration = previousDurations.get(previousDurations.size() - 1);
            previousDurations.remove(previousDurations.size() - 1);
        }


        double[] marginals = new double[maxDuration];

        cp = makeSolver();

        IntVar[] durationsFuture = makeIntVarArray(cp,n - previousDurations.size(), 1, maxDuration);

        try {
            // Rythm constraints
            IntVar violations = rythm(previousDurations, durationsFuture);
            initializeMarginals(durationsFuture, currentDuration, pastDuration);
            //printTime();
            solve();
            //printTime()
            System.out.println(Math.max(0, violations.min()));

            // If there is a contradiction in the constraints, it means that there is no solution
        } catch(InconsistencyException e) {
            printMarginals(marginals);
            //main(null);
        }
    }

    private static IntVar rythm(ArrayList<Integer> previousDurations, IntVar[] durationsFuture) {
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

        IntVar[] violations = makeIntVarArray(cp, durationsFuture.length - 1, 0, 1);
        for (int i = 0; i < durationsFuture.length - 1; i++) {
            boolean isLongI = longNotes[previousDurations.size() + i];
            IntVar[] noteViolations = makeIntVarArray(cp, durationsFuture.length - 1, 0, 1);
            BoolVar previousDurationViolations;
            if(isLongI) {
                previousDurationViolations = isLessOrEqual(durationsFuture[i], maxOfShortNotes);
            } else {
                previousDurationViolations = isLargerOrEqual(durationsFuture[i], minOfLongNotes);
            }

            for (int j = i + 1; j < durationsFuture.length; j++) {
                boolean isLongJ = longNotes[previousDurations.size() + j];
                if(isLongI && !isLongJ) {
                    noteViolations[j - 1] = isLessOrEqual(durationsFuture[i], durationsFuture[j]);
                }
                if(!isLongI && isLongJ) {
                    noteViolations[j - 1] = isLessOrEqual(durationsFuture[j], durationsFuture[i]);
                }
            }
            violations[i] = sum(sum(noteViolations), previousDurationViolations);
        }
        return sum(violations);
    }

    private static void solve() {
        cp.fixPoint();
        cp.beliefPropa();
    }

    private static void initializeMarginals(IntVar[] durations, int currentDuration, int pastDuration) {
        if (pastDuration > 0) {
            durations[0].assign(pastDuration);
            durations[1].removeBelow(currentDuration);
        } else {
            durations[0].removeBelow(currentDuration);
        }
    }

    public static void printMarginals(double[] marginals) {
        int i = 0;
        for (double m : marginals) {
            System.out.println(m);
            i++;
        }
    }

}
