package minicpbp.examples;

import minicpbp.engine.core.BoolVar;
import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.util.exception.InconsistencyException;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static minicpbp.cp.Factory.*;

public class Counterpoint {
    private static Solver cp;
    private static int n = 32;
    private static int tonic = 5;
    private static int dominant = 12;
    private static int nNoteValues = 29;
    private static int minNbCharacteristicModalSkips = 10;
    // Only natural notes + Bb to prevent tritones
    //                                  Bb B  C  D  E  F   G   A   Bb  B   C   D
    private static int[] pitchDomain = {3, 4, 5, 7, 9, 10, 12, 14, 15, 16, 17, 19};
    private static int[] intervalDomain = {1, 2, 3, 4, 5, 7, 8, 9, 12, 0, -1, -2, -3, -4, -5, -7, -8, -9, -12};
    private static int[] sizeOfMotion = {5, 5, 5, 5, 5, 5, 0, 4, 4, 4, 3, 3, 6, 0, 0, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2};
    private static boolean useRandomSequence = false;

    /**
     * Prints the marginals for the next note in the melodic line
     *
     * @param args the array of previous notes
     */
    public static void main(String[] args) {
        // Replaces args with a random sequence if desired
        if (useRandomSequence) {
            Random random = new Random();
            int nNotesInArgs = random.nextInt(n);
            args = new String[nNotesInArgs];
            System.out.print("Args: ");
            for (int i = 0; i < nNotesInArgs; i++) {
                args[i] = random.nextInt(nNoteValues) + "";
                System.out.print(args[i] + " ");
            }
            System.out.println();
        }

        ArrayList<Integer> previousNotes = (ArrayList<Integer>) Arrays.stream(args)
                .map(Integer::parseInt)
                .collect(Collectors.toList());

        double[] marginals = new double[nNoteValues];

        cp = makeSolver();

        // Pitches must be in key (except for previous notes since we have no control)
        IntVar[] pitchesFuture = makeIntVarArray(n - previousNotes.size() + 1,
                (i) -> i == 0 && previousNotes.size() > 0 ? makeIntVar(cp, 0, nNoteValues - 1) :
                        makeIntVar(cp, Arrays.stream(pitchDomain).boxed().collect(Collectors.toSet())));

        // Restricting legal intervals
        IntVar[] intervalsFuture = makeIntVarArray(previousNotes.size() < 2 ? n - 1 : n - previousNotes.size(),
                (i) -> makeIntVar(cp, Arrays.stream(intervalDomain).boxed().collect(Collectors.toSet())));
        IntVar[] intervalsShiftedFuture = makeIntVarArray(cp,
                previousNotes.size() < 2 ? n - 1 : n - previousNotes.size(), 0, 12 - (-12) + 1);
        IntVar[] motionSizeFuture = makeIntVarArray(cp, previousNotes.size() < 2 ? n - 1 : n - previousNotes.size(),
                0, 6);

        // Relationships between arrays
        for (int i = 0; i < motionSizeFuture.length; i++) {
            cp.post(equal(motionSizeFuture[i], elementWithBP(sizeOfMotion, intervalsShiftedFuture[i])));
        }

        for (int i = 0; i < intervalsFuture.length; i++) {
            cp.post(equal(intervalsFuture[i], sum(pitchesFuture[i + 1], minus(pitchesFuture[i]))));
        }

        for (int i = 0; i < intervalsShiftedFuture.length; i++) {
            cp.post(equal(intervalsShiftedFuture[i], sum(intervalsFuture[i], makeIntVar(cp, 12, 12))));
        }

        try {
            // Hard constraints
            tritonOutlines(intervalsShiftedFuture, previousNotes);
            tonicEnds(pitchesFuture);
            stepwiseDescentToFinal(intervalsFuture);
            noRepeat(intervalsFuture);
            coverModalRange(pitchesFuture, previousNotes);
            characteristicModalSkips(pitchesFuture, intervalsFuture, previousNotes);

            // Soft constraints
            BoolVar skipStepsRatio = skipsStepsRatio(intervalsFuture, previousNotes);
            BoolVar sixths = avoidSixths(intervalsFuture);
            BoolVar skipStepsSequence = skipStepsSequence(motionSizeFuture, previousNotes);
            BoolVar bFlat = bFlat(pitchesFuture, intervalsFuture);
            IntVar nBrokenConstraints = sum(skipStepsRatio, sixths, skipStepsSequence, bFlat);
            cp.post(lessOrEqual(nBrokenConstraints, makeIntVar(cp, 0, 0)));

            initializeMarginals(pitchesFuture, previousNotes);
            solve();

            // Obtaining the marginals
            IntVar currentNoteMarginals = pitchesFuture[previousNotes.size() > 0 ? 1 : 0];
            int[] domain = new int[currentNoteMarginals.size()];
            currentNoteMarginals.fillArray(domain);
            for(int value : domain)
                marginals[value] = currentNoteMarginals.marginal(value);
            printMarginals(marginals);

            // If there is a contradiction in the constraints, it means that there is no solution (all marginals are 0)
        } catch(InconsistencyException e) {
            printMarginals(marginals);
        }
    }

    /**
     * Returns a BoolVar indicating if the bFlat constraint is violated
     * A bFlat (pitches 3, 15 or 27) must be followed by a descending interval
     *
     * @param pitches the array of future notes
     * @param intervals the array of future intervals
     * @return a BoolVar indicating if the bFlat constraint is violated
     */
    private static BoolVar bFlat(IntVar[] pitches, IntVar[] intervals) {
        int bFlatPitch = 3;
        IntVar[] bFlat = makeIntVarArray(cp, intervals.length, 0, 1);
        for (int i = 0; i < intervals.length; i++) {
            cp.post(equal(bFlat[i], isEqual(
                    sum(isLarger(
                            sum(
                                    isEqual(pitches[i], bFlatPitch),
                                    isEqual(pitches[i], bFlatPitch + 12),
                                    isEqual(pitches[i], bFlatPitch + 24)),
                            0),
                            isLarger(intervals[i], 0)
                    ), 2)));
        }
        return isLarger(sum(bFlat), 0);
    }

    /**
     * Returns a BoolVar indicating if the skipStepsSequence constraint is violated
     * A skip should be preceded or followed by a step in the opposite direction.
     * Avoid more than two successive skips.
     * Avoid skipping on both sides of a temporary high or low point.
     * Two successive skips in the same direction should be small.
     * Pyramid rule.
     *
     * @param motionSize array indicating the type of motion
     * @param previousNotes the array of previous notes
     * @return a BoolVar indicating if the skipStepsSequence constraint is violated
     */
    private static BoolVar skipStepsSequence(IntVar[] motionSize, ArrayList<Integer> previousNotes) {
        int nMotionSize = 7;
        int mapSize = 6;
        int nbStatesSkip = 1 + mapSize + 4 * mapSize;
        int[][] automaton = new int[nbStatesSkip][nMotionSize];
        int[][] rulePenalties = new int[nbStatesSkip][nMotionSize];
        int[] map = {0, 1, 1, 2, 3, 3};

        for (int i = 0; i < automaton.length; i++) {
            automaton[i][nMotionSize - 1] = i;
        }

        for (int i = 0; i < nMotionSize - 1; i++) {
            automaton[0][i] = 1 + i;
        }
        rulePenalties[0] = new int[]{0, 0, 0, 0, 0, 0, 0};

        for (int i = 1; i < nMotionSize; i++) {
            for (int j = 0; j < mapSize; j++) {
                automaton[i][j] = state(map[i - 1], j);
            }
        }
        rulePenalties[1] = new int[]{0, 1, 1, 0, 0, 0, 0};
        rulePenalties[2] = new int[]{0, 0, 2, 0, 1, 1, 0};
        rulePenalties[3] = new int[]{0, 0, 1, 0, 1, 1, 0};
        rulePenalties[4] = new int[]{0, 0, 0, 0, 0, 0, 0};
        rulePenalties[5] = new int[]{0, 1, 1, 1, 0, 1, 0};
        rulePenalties[6] = new int[]{0, 1, 1, 1, 2, 1, 0};

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < mapSize; j++) {
                for (int k = 0; k < mapSize; k++) {
                    automaton[state(i, j)][k] = state(map[j], k);
                }
            }
        }
        rulePenalties[state(0, 0)] = new int[]{0, 1, 1, 0, 0, 0, 0};
        rulePenalties[state(0, 1)] = new int[]{1, 1, 3, 0, 2, 2, 0};
        rulePenalties[state(0, 2)] = new int[]{1, 1, 2, 0, 2, 2, 0};
        rulePenalties[state(0, 3)] = new int[]{0, 0, 0, 0, 0, 0, 0};
        rulePenalties[state(0, 4)] = new int[]{0, 1, 1, 1, 0, 1, 0};
        rulePenalties[state(0, 5)] = new int[]{0, 1, 1, 1, 2, 1, 0};

        rulePenalties[state(1, 0)] = new int[]{0, 1, 1, 0, 0, 0, 0};
        rulePenalties[state(1, 1)] = new int[]{1, 2, 4, 0, 3, 3, 0};
        rulePenalties[state(1, 2)] = new int[]{1, 2, 3, 0, 3, 3, 0};
        rulePenalties[state(1, 3)] = new int[]{0, 0, 0, 0, 0, 0, 0};
        rulePenalties[state(1, 4)] = new int[]{0, 3, 3, 2, 2, 3, 0};
        rulePenalties[state(1, 5)] = new int[]{0, 3, 3, 2, 4, 3, 0};

        rulePenalties[state(2, 0)] = new int[]{0, 1, 1, 0, 0, 0, 0};
        rulePenalties[state(2, 1)] = new int[]{0, 0, 2, 0, 1, 1, 0};
        rulePenalties[state(2, 2)] = new int[]{0, 1, 1, 0, 1, 1, 0};
        rulePenalties[state(2, 3)] = new int[]{0, 0, 0, 0, 0, 0, 0};
        rulePenalties[state(2, 4)] = new int[]{0, 2, 2, 2, 1, 2, 0};
        rulePenalties[state(2, 5)] = new int[]{0, 2, 2, 2, 1, 2, 0};

        rulePenalties[state(3, 0)] = new int[]{0, 1, 1, 0, 0, 0, 0};
        rulePenalties[state(3, 1)] = new int[]{1, 2, 4, 0, 3, 3, 0};
        rulePenalties[state(3, 2)] = new int[]{1, 2, 3, 0, 3, 3, 0};
        rulePenalties[state(3, 3)] = new int[]{0, 0, 0, 0, 0, 0, 0};
        rulePenalties[state(3, 4)] = new int[]{0, 3, 3, 2, 2, 3, 0};
        rulePenalties[state(3, 5)] = new int[]{0, 3, 3, 2, 3, 3, 0};

        // We adjust the beginning state and the cost allowed with the previous notes
        int maxCost = 0;
        int state = 0;
        for (int i = 1; i < previousNotes.size(); i++) {
            int intervalShifted = previousNotes.get(i) - previousNotes.get(i - 1) + 12;
            intervalShifted = Math.min(Math.max(intervalShifted, 0), 24);
            int motion = sizeOfMotion[intervalShifted];
            state = automaton[state][motion];
            maxCost -= rulePenalties[state][motion];
        }

        IntVar cost = makeIntVar(cp, 0, 100000);
        cp.post(costRegular(motionSize, automaton, state,
                IntStream.rangeClosed(0, nbStatesSkip - 1).boxed().collect(Collectors.toList()), rulePenalties, cost));

        return isLarger(cost, Math.max(0, maxCost));
    }

    /**
     * Computes next state based on two consecutive motions (used in skipStepsSequence)
     *
     * @param motion1 first motion
     * @param motion2 second motion
     * @return an int representing the next state
     */
    private static int state(int motion1, int motion2) {
        return 7 + 6 * motion1 + motion2;
    }

    /**
     * Returns a BoolVar indicating if the avoidSixths constraint is violated
     * We want to avoid intervals of a sixth, except for an ascending minor sixth
     *
     * @param intervals array indicating the future intervals
     * @return a BoolVar indicating if the avoidSixths constraint is violated
     */
    private static BoolVar avoidSixths(IntVar[] intervals) {
        IntVar[] sixths = makeIntVarArray(cp, intervals.length, 0, 1);
        for (int i = 0; i < intervals.length; i++) {
            cp.post(equal(
                    sixths[i],
                    isLarger(
                            sum(
                                    isEqual(intervals[i], -9),
                                    isEqual(intervals[i], -8),
                                    isEqual(intervals[i], 9)
                            ), 0
                    )
            ));
        }

        return isLarger(sum(sixths), 0);
    }

    /**
     * Returns a BoolVar indicating if the skipsStepsRatio constraint is violated
     * Use more steps (small intervals) than skips (large intervals)
     *
     * @param intervals array indicating the future intervals
     * @param previousNotes array indicating the previous notes
     * @return a BoolVar indicating if the sipStepsRatio constraint is violated
     */
    private static BoolVar skipsStepsRatio(IntVar[] intervals, ArrayList<Integer> previousNotes) {
        int maxInterval = 28;
        IntVar[] intervalsShifted = makeIntVarArray(intervals.length,
                i -> sum(intervals[i], makeIntVar(cp, maxInterval, maxInterval)));

        int[] skipMotion = new int[maxInterval * 2 + 1];
        for (int i = -maxInterval; i <= maxInterval; i++)
            skipMotion[28 + i] = 1;

        skipMotion[maxInterval + -2] = 0;
        skipMotion[maxInterval + -1] = 0;
        skipMotion[maxInterval + 0] = 0;
        skipMotion[maxInterval + 1] = 0;
        skipMotion[maxInterval + 2] = 0;

        int nSkips = 0;
        for (int i = 1; i < previousNotes.size(); i++) {
            int interval = previousNotes.get(i) - previousNotes.get(i - 1);
            int intervalShifted = interval + maxInterval;
            if (skipMotion[intervalShifted] == 1)
                nSkips++;
        }

        IntVar[] skips = makeIntVarArray(cp, intervalsShifted.length, 0, 1);
        for (int i = 0; i < intervalsShifted.length; i++) {
            cp.post(equal(skips[i], elementWithBP(skipMotion, intervalsShifted[i])));
        }

        return isLargerOrEqual(sum(skips), (n - 1)/2 - nSkips);
    }

    /**
     * Enforces the characteristicModalSkips constraint
     * We want a minimum of three intervals between C and G or G and C
     * C: 5, 17
     * G: 12
     *
     * @param pitches array indicating the future notes
     * @param intervals array indication the future intervals
     * @param previousNotes array indicating the previous notes
     */
    private static void characteristicModalSkips(IntVar[] pitches, IntVar[] intervals, ArrayList<Integer> previousNotes) {
        // We compute characteristicModalSkips that are already there and adjust the number needed
        if (previousNotes.size() > 1) {
            for (int i = 0; i < previousNotes.size() - 1; i++) {
                int note = previousNotes.get(i);
                int interval = previousNotes.get(i + 1) - note;
                if ((note == tonic && interval == 7) || (note == tonic + 12 && interval == -5) ||
                        (note == dominant && interval == -7) || (note == dominant && interval == 5))
                    minNbCharacteristicModalSkips--;
            }
        }

        // Constraint for the rest of the notes
        IntVar[] isSkip = makeIntVarArray(cp, intervals.length, 0, 1);
        for (int i = 0; i < isSkip.length; i++) {
            BoolVar isTonic7 = isEqual(sum(isEqual(pitches[i], tonic), isEqual(intervals[i], 7)), 2);
            BoolVar isTonic5 = isEqual(sum(isEqual(pitches[i], tonic + 12), isEqual(intervals[i], -5)), 2);
            BoolVar isDominant7 = isEqual(sum(isEqual(pitches[i], dominant), isEqual(intervals[i], -7)), 2);
            BoolVar isDominant5 = isEqual(sum(isEqual(pitches[i], dominant), isEqual(intervals[i], 5)), 2);

            cp.post(equal(isSkip[i], isLarger(sum(isTonic7, isTonic5, isDominant7, isDominant5), 0)));
        }

        cp.post(largerOrEqual(sum(isSkip), makeIntVar(cp, minNbCharacteristicModalSkips, minNbCharacteristicModalSkips)));


    }

    /**
     * Enforces the coverModalRange constraint
     * We want the notes in the range {5, 7, 9, 10, 12, 14, 16, 17, 19} to occur at least once
     *
     * @param pitches array indicating the future notes
     * @param previousNotes array indicating the previous notes
     */
    private static void coverModalRange(IntVar[] pitches, ArrayList<Integer> previousNotes) {
        int[] pitchesToRestrict = new int[]{5, 7, 9, 10, 12, 14, 16, 17, 19};
        IntVar[] pitchOccurrences = makeIntVarArray(cp, 9, 0, n);

        IntVar[] pitchesFuture = makeIntVarArray(previousNotes.size() > 0 ? pitches.length - 1 : pitches.length,
                i -> previousNotes.size() > 0 ? pitches[i + 1] : pitches[i]);
        cp.post(cardinality(pitchesFuture, pitchesToRestrict, pitchOccurrences));

        HashMap<Integer, Integer> pastOccurrences = new HashMap<>();
        for (int pitch : pitchesToRestrict)
            pastOccurrences.put(pitch, 0);

        for (Integer note : previousNotes) {
            if (pastOccurrences.containsKey(note)) {
                pastOccurrences.put(note, pastOccurrences.get(note) + 1);
            }
        }

        for (int i = 0; i < pitchesToRestrict.length; i++) {
            if (pastOccurrences.get(pitchesToRestrict[i]) == 0) {
                cp.post(largerOrEqual(pitchOccurrences[i], makeIntVar(cp, 1, 1)));
            }
        }

    }

    /**
     * Enforces the noRepeat constraint
     * We forbid intervals of 0
     *
     * @param intervalsFuture array indicating the future intervals
     */
    private static void noRepeat(IntVar[] intervalsFuture) {

        Arrays.stream(intervalsFuture).forEach(i -> cp.post(notEqual(i, makeIntVar(cp, 0, 0))));
    }

    /**
     * Enforces the stepwiseDescentToFinal constraint
     * We want the last interval to be either -2 or -1
     *
     * @param intervals array indicating the future intervals
     */
    private static void stepwiseDescentToFinal(IntVar[] intervals) {
        cp.post(largerOrEqual(intervals[intervals.length - 1], makeIntVar(cp, -2, -2)));
        cp.post(lessOrEqual(intervals[intervals.length - 1], makeIntVar(cp, -1, -1)));
    }

    /**
     * Enforces the tonicEnds constraint
     * We want the last note to be the tonic
     *
     * @param pitches array indicating the future notes
     */
    private static void tonicEnds(IntVar[] pitches) {
        cp.post(equal(pitches[pitches.length - 1], makeIntVar(cp, tonic, tonic)));
    }

    /**
     * Enforces the tritonOutlines constraint
     * An outline of an augmented fourth is prohibited.
     * An outline of a diminished fifth is allowed only if it is completely filled in by step (interval smaller than
     * 2) and then followed by a step in the opposite direction.
     *
     * @param intervalsShiftedFuture array indicating the future intervals (shifted between 0 and 24)
     * @param previousNotes array indicating the previous notes
     */
    private static void tritonOutlines(IntVar[] intervalsShiftedFuture, ArrayList<Integer> previousNotes) {
        int nbStates = 1 + 2 * 13 + 1;
        int error = 27;
        int beginningState = 0;
        int middle = 12;
        int descending = 13;
        ArrayList<Integer> goalStates = new ArrayList<>();
        for (int i = 1; i < 9; i ++) {
            goalStates.add(i);
        }

        goalStates.add(10);

        for (int i = 12; i < 22; i++) {
            goalStates.add(i);
        }

        goalStates.addAll(Arrays.asList(23, 25, 26));

        int[][] automaton = new int[nbStates][12 - (-12) + 1];

        //If the note is repeated, we do not change the state
        for( int i=0; i<automaton.length; i++)
            automaton[i][middle+0] = i;

        // INITIAL STATE
        for(int j=1; j<=5; j++) {
            automaton[0][middle+j] = j; // start ascending outline
            automaton[0][middle-j] = descending+j; // start descending outline
        }
        for(int j = 6 ; j<=middle; j++) {
            automaton[0][middle+j] = descending; // ascending outline beyond tritone
            automaton[0][middle-j] = descending+13; // descending outline beyond tritone
        }
        // 2-NOTES STATES
        // states 1 and 13+1
        automaton[1][middle+1] = error; // no transition on two consecutive semi-tones
        automaton[descending+1][middle-1] = error; // no transition on two consecutive semi-tones
        for(int j=2; j<=5; j++) { // continue with outline
            automaton[1][middle+j] = 4+j;
            automaton[descending+1][middle-j] = descending+4+j;
        }
        for(int j = 6 ; j<=middle; j++) { // outline beyond tritone
            automaton[1][middle+j] = descending;
            automaton[descending+1][middle-j] = descending+13;
        }
        for(int j=1; j<=5; j++) {
            automaton[descending+1][middle+j] = j; // start ascending outline
            automaton[1][middle-j] = descending+j; // start descending outline
        }
        for(int j=7; j<=middle; j++) {
            automaton[descending+1][middle+j] = descending; // ascending outline beyond tritone
            automaton[1][middle-j] = descending+13; // descending outline beyond tritone
        }
        // other 2-notes states
        for(int i=2; i<=5; i++) {
            for(int j=1; j<=6-i; j++) { // continue with outline
                automaton[i][middle+j] = i+3+j;
                automaton[descending+i][middle-j] = descending+i+3+j;
            }
            for(int j = 7 ; j<=middle; j++) { // outline beyond tritone
                automaton[i][middle+j] = descending;
                automaton[descending+i][middle-j] = descending+13;
            }
            for(int j=1; j<=5; j++) {
                automaton[descending+i][middle+j] = j; // start ascending outline
                automaton[i][middle-j] = descending+j; // start descending outline
            }
            for(int j=7; j<=middle; j++) {
                automaton[descending+i][middle+j] = descending; // ascending outline beyond tritone
                automaton[i][middle-j] = descending+13; // descending outline beyond tritone
            }
        }
        // 3-NOTES STATES
        // states 6 and 13+6
        automaton[6][middle+1] = error; // no transition on too close semi-tones
        automaton[descending+6][middle-1] = error; // no transition on too close semi-tones
        for(int j=2; j<=3; j++) { // continue with outline
            automaton[6][middle+j] = 8+j;
            automaton[descending+6][middle-j] = descending+8+j;
        }
        for(int j = 4 ; j<=middle; j++) { // outline beyond tritone
            automaton[6][middle+j] = descending;
            automaton[descending+6][middle-j] = descending+13;
        }
        for(int j=1; j<=5; j++) {
            automaton[descending+6][middle+j] = j; // start ascending outline
            automaton[6][middle-j] = descending+j; // start descending outline
        }
        for(int j=7; j<=middle; j++) {
            automaton[descending+6][middle+j] = descending; // ascending outline beyond tritone
            automaton[6][middle-j] = descending+13; // descending outline beyond tritone
        }
        // other 3-notes states
        for(int i=7; i<=9; i++) {
            for(int j=1; j<=9-i; j++) { // continue with outline
                automaton[i][middle+j] = i+2+j;
                automaton[descending+i][middle-j] = descending+i+2+j;
            }
            for(int j = 10 ; j<=middle; j++) { // outline beyond tritone
                automaton[i][middle+j] = descending;
                automaton[descending+i][middle-j] = descending+13;
            }
            for(int j=1; j<=5; j++) {
                automaton[descending+i][middle+j] = j; // start ascending outline
                automaton[i][middle-j] = descending+j; // start descending outline
            }
            for(int j=7; j<=middle; j++) {
                automaton[descending+i][middle+j] = descending; // ascending outline beyond tritone
                automaton[i][middle-j] = descending+13; // descending outline beyond tritone
            }
        }
        // but actually states 9 and 13+9 cannot end an outline so..
        for(int j=1; j<=middle; j++) {
            automaton[descending+9][middle+j] = error;
            automaton[9][middle-j] = error;
        }
        // 4-NOTES STATES
        // states 10 and 13+10
        automaton[10][middle+1] = middle; // continue with outline
        automaton[descending+10][middle-1] = descending+12;
        for(int j=2; j<=middle; j++) { // outline beyond tritone
            automaton[10][middle+j] = descending;
            automaton[descending+10][middle-j] = descending+13;
        }
        for(int j=1; j<=5; j++) {
            automaton[descending+10][middle+j] = j; // start ascending outline
            automaton[10][middle-j] = descending+j; // start descending outline
        }
        for(int j=7; j<=middle; j++) {
            automaton[descending+10][middle+j] = descending; // ascending outline beyond tritone
            automaton[10][middle-j] = descending+13; // descending outline beyond tritone
        }
        // states 11 and 13+11
        for(int j=1; j<=middle; j++) { // outline beyond tritone
            automaton[11][middle+j] = descending;
            automaton[descending+11][middle-j] = descending+13;
        }
        for(int j=1; j<=middle; j++) {
            automaton[descending+11][middle+j] = error;
            automaton[11][middle-j] = error;
        }
        // 5-NOTES STATES
        // states 12 and 13+12
        for(int j=1; j<=middle; j++) { // outline beyond tritone
            automaton[middle][middle+j] = descending;
            automaton[descending+12][middle-j] = descending+13;
        }
        for(int j=1; j<=2; j++) {
            automaton[descending+12][middle+j] = j;
            automaton[middle][middle-j] = descending+j;
        }
        for(int j=3; j<=middle; j++) {
            automaton[descending+12][middle+j] = error;
            automaton[middle][middle-j] = error;
        }
        // "BEYOND" STATES
        for(int j=1; j<=middle; j++) { // outline beyond tritone
            automaton[descending][middle+j] = descending;
            automaton[descending+13][middle-j] = descending+13;
        }
        for(int j=1; j<=5; j++) {
            automaton[descending+13][middle+j] = j; // start ascending outline
            automaton[descending][middle-j] = descending+j; // start descending outline
        }
        for(int j=7; j<=middle; j++) {
            automaton[descending+13][middle+j] = descending; // ascending outline beyond tritone
            automaton[descending][middle-j] = descending+13; // descending outline beyond tritone
        }

        for (int i = 0; i < 25; i++) {
            automaton[error][i] = error;
        }

        // Figuring out beginning state:
        int state = beginningState;
        for(int i = 1; i < previousNotes.size(); i++) {
            int interval = previousNotes.get(i) - previousNotes.get(i - 1);
            int intervalShifted = interval + middle;
            // If the interval is not valid, we clip it
            intervalShifted = Math.min(Math.max(intervalShifted, 0), 24);
            state = automaton[state][intervalShifted];

            // When we reach an error state, instead of being stuck there, we reset
            if (state == error)
                state = beginningState;
        }
        beginningState = state;

        cp.post(regular(intervalsShiftedFuture, automaton, beginningState, goalStates));
    }

    /**
     * Propagates the constraints and computes the marginals
     */
    private static void solve() {
        cp.fixPoint();
        cp.beliefPropa();
    }

    /**
     * Assigns the last chosen note to its value
     *
     * @param pitches array indicating the relevant notes
     * @param previousNotes array indicating the previous notes
     */
    private static void initializeMarginals(IntVar[] pitches, ArrayList<Integer> previousNotes) {
        // We set the last previous note (not the current) to the chosen value
        if (previousNotes.size() > 0) {
            pitches[0].assign(previousNotes.get(previousNotes.size() - 1));
        }
    }

    /**
     * Outputs the marginals
     *
     * @param marginals array indicating the marginal probability for each pitch value
     */
    private static void printMarginals(double[] marginals) {
        for (double m : marginals) {
            System.out.println(m);
        }
    }

}
