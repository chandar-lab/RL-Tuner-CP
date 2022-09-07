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

public class RLTunerRythmToken {
    private static Solver cp;
    private static int n = 14;
    private static boolean useRandomSequence = false;
    private static int maxDuration = 56;
    private static int[] rythmDomain = {1, 2, 4};

    public static void main(String[] args) {
        int sumDurations = args.length;
        ArrayList<Integer> previousDurations = new ArrayList<>();
        ArrayList<Integer> previousNotes = new ArrayList<>();
        int index = 0;
        while(index < args.length) {
            int note = Integer.parseInt(args[index]);
            int duration = 0;
            while (index < args.length && note == Integer.parseInt(args[index])) {
                duration++;
                index++;
            }
            previousDurations.add(duration);
            previousNotes.add(note);
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
            sumDurations = previousDurations.stream().mapToInt(a->a).sum();
        }
        /*
        Si on décide de changer de note, il faut prendre en compte la durée finale de la dernière note: ça compte comme une violation potentielle
        Si on décide de prolonger la note, il faut prendre en compte la durée de cette note
         */
        int currentDuration = previousDurations.get(previousDurations.size() - 1);
        sumDurations -= currentDuration;
        previousDurations.remove(previousDurations.size() - 1);
        previousNotes.remove(previousNotes.size() - 1);
        int pastDuration = 0;
        if (previousDurations.size() > 0 && currentDuration == 1) {
            pastDuration = previousDurations.get(previousDurations.size() - 1);
            sumDurations -= previousDurations.get(previousDurations.size() - 1);
            previousDurations.remove(previousDurations.size() - 1);
            previousNotes.remove(previousNotes.size() - 1);
        }

        cp = makeSolver();
        int size = n - previousDurations.size();
        IntVar[] durationsFuture = makeIntVarArray(cp, size, 1, maxDuration);

        try {
            initializeMarginals(durationsFuture, currentDuration, pastDuration);
            // Rythm constraints
            //IntVar violations = rythm(previousDurations, durationsFuture);
            IntVar smallViolations = smallDurations(previousDurations, durationsFuture);
            IntVar naturalRythmViolations = naturalRythm(durationsFuture);
            IntVar maxTokensViolations = maxTokens(sumDurations, durationsFuture);

            //printTime();
            //solve();
            //printTime()
            //System.out.println(Math.max(0, violations.min()));
            System.out.println(Math.max(0, naturalRythmViolations.min()));
            System.out.println(Math.max(0, maxTokensViolations.min()));
            System.out.println(Math.max(0, smallViolations.min()));


            // If there is a contradiction in the constraints, it means that there is no solution
        } catch(InconsistencyException e) {
            System.out.println("Should not go there");
        }
    }

    private static IntVar smallDurations(ArrayList<Integer> previousDurations, IntVar[] durationsFuture) {
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
        cp.post(cardinality(durationsFuture, new int[]{1, 2, 4}, occurrences));

        IntVar violations42 = maximum(makeIntVar(cp, 0, 0), sum(sum(occurrences[2], makeIntVar(cp, durationFour, durationFour)), minus(sum(occurrences[1], makeIntVar(cp, durationTwo, durationTwo)))));
        IntVar violations21 = maximum(makeIntVar(cp, 0, 0), sum(sum(occurrences[1], makeIntVar(cp, durationTwo, durationTwo)), minus(sum(occurrences[0], makeIntVar(cp, durationOne, durationOne)))));
        return sum(violations21, violations42);
    }

    private static IntVar maxTokens(int sumDurations, IntVar[] durationsFuture) {
        int remainingTokens = maxDuration - sumDurations;
        return minus(sum(durationsFuture), remainingTokens);
    }

    private static IntVar naturalRythm(IntVar[] durationsFuture) {
        IntVar[] notNaturalDuration = makeIntVarArray(cp, durationsFuture.length, 0, 1);
        for (int i = 0; i < durationsFuture.length; i++) {
            notNaturalDuration[i] = isEqual(sum(isEqual(durationsFuture[i], 1), isEqual(durationsFuture[i], 2), isEqual(durationsFuture[i], 4)), makeIntVar(cp, 0, 0));
        }

        IntVar[] exceedsMax = makeIntVarArray(cp, durationsFuture.length, 0, maxDuration);
        for (int i = 0; i < durationsFuture.length; i++) {
            exceedsMax[i] = maximum(makeIntVar(cp, 0, 0), sum(durationsFuture[i], makeIntVar(cp, -4, -4)));
        }
        return sum(sum(notNaturalDuration), sum(exceedsMax));
    }

    private static IntVar rythm(ArrayList<Integer> previousDurations, IntVar[] durationsFuture) {
        int maxOfShortNotes = 0;
        int minOfLongNotes = 1000;
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

        ArrayList<Integer> longNotes = new ArrayList<>(Arrays.asList(
                1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1));



        IntVar[] violations = makeIntVarArray(cp, durationsFuture.length, 0, 1);
        for (int i = 0; i < durationsFuture.length; i++) {
            int statusI = longNotes.get(previousDurations.size() + i);
            IntVar[] noteViolations = makeIntVarArray(cp, durationsFuture.length - 1, 0, 1);
            IntVar previousDurationViolations;
            if(statusI == 2) {
                previousDurationViolations = maximum(makeIntVar(cp, 0, 0), sum(makeIntVar(cp, maxOfShortNotes + 1, maxOfShortNotes + 1), minus(durationsFuture[i])));
            } else if (statusI == 1){
                previousDurationViolations = maximum(makeIntVar(cp, 0, 0), sum(makeIntVar(cp, -minOfLongNotes + 1, -minOfLongNotes + 1), durationsFuture[i]));
            } else {
                previousDurationViolations = makeIntVar(cp, 0, 0);
            }
            if (i != durationsFuture.length - 1) {
                for (int j = i + 1; j < durationsFuture.length; j++) {
                    int statusJ = longNotes.get(previousDurations.size() + j);
                    if (statusI == 2 && statusJ == 1) {
                        noteViolations[j - 1] = maximum(makeIntVar(cp, 0, 0), sum(durationsFuture[j], minus(durationsFuture[i])));
                    }
                    if (statusI == 1 && statusJ == 2) {
                        noteViolations[j - 1] = maximum(makeIntVar(cp, 0, 0), sum(durationsFuture[i], minus(durationsFuture[j])));
                    }
                }
            }
            violations[i] = sum(durationsFuture.length > 1 ? sum(noteViolations) : makeIntVar(cp, 0, 0), previousDurationViolations);
        }
        return sum(violations);
    }

    private static void solve() {
        cp.fixPoint();
    }

    private static void initializeMarginals(IntVar[] durations, int currentDuration, int pastDuration) {
        if (pastDuration > 0) {
            durations[0].assign(pastDuration);
            durations[1].removeBelow(currentDuration);
        } else {
            durations[0].removeBelow(currentDuration);
        }
    }

}
