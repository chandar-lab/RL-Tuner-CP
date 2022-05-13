package minicpbp.examples;

import minicpbp.engine.core.BoolVar;
import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.util.exception.InconsistencyException;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static minicpbp.cp.Factory.*;

public class CounterpointViolation {
    private static Solver cp;
    private static int n = 32;
    private static int noteMax = 28;
    private static int tonic = 5;
    private static int dominant = 12;
    private static int minNbCharacteristicModalSkips = 3;
    private static int[] intervalDomain = {1, 2, 3, 4, 5, 7, 8, 9, 12, 0, -1, -2, -3, -4, -5, -7, -8, -9, -12};
    private static int[] sizeOfMotion = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 5, 5, 5, 5, 5, 0, 4, 4, 4, 3, 3, 6, 0, 0, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    private static boolean useRandomSequence = false;

    /**
     * Prints the number of violations associated to the current note in the melodic line for each constraint
     *
     * @param args the array of previous and current notes
     */
    public static void main(String[] args) {
        // Replaces args with a random sequence if desired
        if (useRandomSequence) {
            Random random = new Random();
            int nNotesInArgs = random.nextInt(n) + 1;
            args = new String[nNotesInArgs];
            System.out.print("Args: ");
            for (int i = 0; i < nNotesInArgs; i++) {
                args[i] = random.nextInt(noteMax + 1) + "";
                System.out.print(args[i] + " ");
            }
            System.out.println();
        }

        ArrayList<Integer> previousNotes = (ArrayList<Integer>) Arrays.stream(args)
                .map(Integer::parseInt)
                .collect(Collectors.toList());

        int presentNote = previousNotes.get(previousNotes.size() - 1);
        previousNotes.remove(previousNotes.size() - 1);

        cp = makeSolver();

        // We cannot enforce the pitches to have valid values. So we take the whole range
        IntVar[] pitchesFuture = makeIntVarArray(cp, n - previousNotes.size() + 1, 0, noteMax);
        IntVar[] intervalsFuture = makeIntVarArray(previousNotes.size() < 2 ? n - 1 : n - previousNotes.size(),
                (i) -> previousNotes.size() > 0 && i == 0 ?
                        makeIntVar(cp, -noteMax, noteMax) :
                        makeIntVar(cp, Arrays.stream(intervalDomain).boxed().collect(Collectors.toSet())));
        IntVar[] intervalsShiftedFuture = makeIntVarArray(cp,
                previousNotes.size() < 2 ? n - 1 : n - previousNotes.size(), -noteMax + 12, noteMax + 12);
        IntVar[] motionSizeFuture = makeIntVarArray(cp,
                previousNotes.size() < 2 ? n - 1 : n - previousNotes.size(), -1, 6);

        // Relationships between arrays
        for (int i = 0; i < motionSizeFuture.length; i++) {
            cp.post(equal(motionSizeFuture[i], elementWithBP(sizeOfMotion,
                    sum(intervalsFuture[i], makeIntVar(cp, noteMax, noteMax)))));
        }

        for (int i = 0; i < intervalsFuture.length; i++) {
            cp.post(equal(intervalsFuture[i], sum(pitchesFuture[i + 1], minus(pitchesFuture[i]))));
        }

        for (int i = 0; i < intervalsShiftedFuture.length; i++) {
            cp.post(equal(intervalsShiftedFuture[i], sum(intervalsFuture[i], makeIntVar(cp, 12, 12))));
        }

        initializePitches(pitchesFuture, previousNotes, presentNote);

        try {
            // Hard constraints
            IntVar naturalNotes = naturalNotes(pitchesFuture, previousNotes);
            BoolVar intervals = intervals(previousNotes, presentNote);
            // If the interval constraint is not violated, we can assume that the current interval is between 0 and 24,
            // that it is not 6 or 18 and that the motion size is not -1
            // This restriction allows us to apply the regular constraints without change
            if (intervals.isFalse()) {
                intervalsShiftedFuture[0].removeAbove(24);
                intervalsShiftedFuture[0].removeBelow(0);
                intervalsShiftedFuture[0].remove(6);
                intervalsShiftedFuture[0].remove(18);
                motionSizeFuture[0].remove(-1);
            }
            // We assume 1 violation if the interval is not valid
            IntVar tritonOutlines = (intervals.isFalse() ?
                    tritonOutlines(intervalsShiftedFuture, previousNotes) : makeIntVar(cp, 1, 1));
            BoolVar tonicEnds = tonicEnds(pitchesFuture);
            BoolVar stepwiseDescentToFinal = stepwiseDescentToFinal(intervalsFuture);
            IntVar repeats = noRepeat(intervalsFuture);
            IntVar coverModalRange = coverModalRange(pitchesFuture, previousNotes);
            IntVar characteristicModalSkips = characteristicModalSkips(pitchesFuture, intervalsFuture, previousNotes);

            // Soft constraints
            IntVar skipStepsRatio = skipsStepsRatio(intervalsFuture, previousNotes);
            IntVar sixths = avoidSixths(intervalsFuture);
            IntVar skipStepsSequence = skipStepsSequence(motionSizeFuture, previousNotes, intervals, presentNote);
            IntVar bFlat = bFlat(pitchesFuture, intervalsFuture);
            solve();

            if (!useRandomSequence)
                printViolations(new IntVar[]{naturalNotes, intervals, tritonOutlines, tonicEnds, stepwiseDescentToFinal,
                    repeats, coverModalRange, characteristicModalSkips, skipStepsRatio, sixths, skipStepsSequence, bFlat});
            // If there is a contradiction in the constraints, it means that there is no solution
        } catch(InconsistencyException e) {
            System.out.println("Should not go there");
        }
    }

    /**
     * Prints the minimum number of violations
     *
     * @param violations array containing the number of violations
     */
    private static void printViolations(IntVar[] violations) {
        for (IntVar violation : violations)
            System.out.println(Math.max(0, violation.min()));
    }

    /**
     * Returns a BoolVar indicating if the intervals constraint is violated
     * The interval must be either {-12, -9, -8, -7, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 8, 9, 12}
     *
     * @param previousNotes the array of previous notes
     * @param presentNote the value of the current note
     *
     * @return a BoolVar indicating if the intervals constraint is violated
     */
    private static BoolVar intervals(ArrayList<Integer> previousNotes, int presentNote) {
        BoolVar intervalViolation = makeBoolVar(cp);
        if (previousNotes.size() == 0) {
            intervalViolation.assign(false);
            return intervalViolation;
        }
        int interval = Math.abs(presentNote - previousNotes.get(previousNotes.size() - 1));
        //{1, 2, 3, 4, 5, 7, 8, 9, 12, 0, -1, -2, -3, -4, -5, -7, -8, -9, -12}
        if (interval > 12 || interval == 6 || (interval > 9 && interval < 12)) {
            intervalViolation.assign(true);
        } else {
            intervalViolation.assign(false);
        }
        return intervalViolation;
    }

    /**
     * Returns a IntVar indicating the number of notes violating the naturalNotes constraint
     * The pitch must be either {0, 2, 3, 4, 5, 7, 9, 10, 12, 14, 15, 16, 17, 19, 21, 22, 24, 26, 27, 28}
     *
     * @param pitches the array of future notes
     * @param previousNotes the array of previous notes
     *
     * @return a IntVar indicating the number of notes violating the naturalNotes constraint
     */
    private static IntVar naturalNotes(IntVar[] pitches, ArrayList<Integer> previousNotes) {
        // 1, 6, 8, 11, 13, 18, 20, 23, 25
        IntVar[] pitchesFuture = makeIntVarArray(previousNotes.size() > 0 ? pitches.length - 1 : pitches.length,
                i -> previousNotes.size() > 0 ? pitches[i + 1] : pitches[i]);

        IntVar[] notNaturalNoteOcc = makeIntVarArray(cp, 9, 0, pitchesFuture.length);
        cp.post(cardinality(pitchesFuture, new int[]{1, 6, 8, 11, 13, 18, 20, 23, 25}, notNaturalNoteOcc));

        return sum(notNaturalNoteOcc);
    }

    /**
     * Returns a IntVar indicating the number of intervals violating the bFlat constraint
     * A bFlat (pitches 3, 15 or 27) must be followed by a descending interval
     *
     * @param pitches the array of future notes
     * @param intervals the array of future intervals
     *
     * @return a IntVar indicating the number of intervals violating the bFlat constraint
     */
    private static IntVar bFlat(IntVar[] pitches, IntVar[] intervals) {
        int bFlatValue = 3;
        int octave = 12;
        IntVar[] bFlat = makeIntVarArray(cp, intervals.length, 0, 1);
        for (int i = 0; i < intervals.length; i++) {
            cp.post(equal(bFlat[i], isEqual(
                    sum(isLarger(sum(isEqual(pitches[i], bFlatValue), isEqual(pitches[i], bFlatValue + octave),
                            isEqual(pitches[i], bFlatValue + octave + octave)), 0),
                            isLarger(intervals[i], 0)
                    ), 2)));
        }
        return sum(bFlat);
    }

    /**
     * Returns a IntVar indicating the number of intervals violating the skipStepsSequence constraint
     * A bFlat (pitches 3, 15 or 27) must be followed by a descending interval
     *
     * @param motionSize the array indicating the type of intervals
     * @param previousNotes the array of previous notes
     * @param invalidLastInterval variable indicating if the last interval is valid
     *                            (will change the behavior of the constraint)
     * @param presentNote value of the current note
     *
     * @return a IntVar indicating the number of intervals violating the skipStepsSequence constraint
     */
    private static IntVar skipStepsSequence(IntVar[] motionSize, ArrayList<Integer> previousNotes, BoolVar invalidLastInterval, int presentNote) {
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
        int state = 0;
        for (int i = 1; i < previousNotes.size(); i++) {
            int intervalShifted = previousNotes.get(i) - previousNotes.get(i - 1) + 28;
            intervalShifted = Math.min(Math.max(intervalShifted, 16), 40);
            int motion = sizeOfMotion[intervalShifted];
            state = automaton[state][motion];
        }

        if (invalidLastInterval.isTrue()) {
            // If the last interval is not valid, we take the average cost at the current state
            int[] stateCosts = rulePenalties[state];
            int[] motionOccurence = new int[]{4, 3, 6, 2, 3, 6, 1};
            int c = 0;
            for (int i = 0; i < stateCosts.length; i++) {
                c += stateCosts[i] * motionOccurence[i];
            }
            c /= 25;

            // If the sequence is not done, we apply the automaton to the future intervals to compute their cost
            if (motionSize.length > 1) {
                int intervalShifted = presentNote - previousNotes.get(previousNotes.size() - 1) + 28;
                intervalShifted = Math.min(Math.max(intervalShifted, 16), 40);
                int motion = sizeOfMotion[intervalShifted];
                int newState = automaton[state][motion];

                IntVar[] motionSizeFuture = makeIntVarArray(motionSize.length - 1, i -> motionSize[i + 1]);
                IntVar cost = makeIntVar(cp, 0, 4 * motionSize.length);
                cp.post(costRegular(motionSizeFuture, automaton, newState,
                        IntStream.rangeClosed(0, nbStatesSkip - 1).boxed().collect(Collectors.toList()), rulePenalties, cost));

                return sum(cost, makeIntVar(cp, c, c));
            }
            return makeIntVar(cp, c, c);
        }

        IntVar cost = makeIntVar(cp, 0, 4 * motionSize.length);
        cp.post(costRegular(motionSize, automaton, state,
                IntStream.rangeClosed(0, nbStatesSkip - 1).boxed().collect(Collectors.toList()), rulePenalties, cost));

        return cost;
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
     * Returns a IntVar indicating the number of intervals violating the avoidSixths constraint
     * We want to avoid intervals of a sixth, except for an ascending minor sixth
     *
     * @param intervals array indicating the future intervals
     * @return a IntVar indicating the number of violations
     */
    private static IntVar avoidSixths(IntVar[] intervals) {
        IntVar[] sixths = makeIntVarArray(cp, intervals.length, 0, 1);
        for (int i = 0; i < intervals.length; i++) {
            cp.post(equal(
                    sixths[i],
                    isLarger(
                            sum(isEqual(intervals[i], -9), isEqual(intervals[i], -8), isEqual(intervals[i], 9)),
                            0
                    )
            ));
        }

        return sum(sixths);
    }
    /**
     * Returns a IntVar indicating the number of intervals violating the skipStepsRatio constraint
     * Use more steps (small intervals) than skips (large intervals)
     *
     * @param intervals array indicating the future intervals
     * @param previousNotes array indicating the previous notes
     * @return a IntVar indicating the number of violations
     */
    private static IntVar skipsStepsRatio(IntVar[] intervals, ArrayList<Integer> previousNotes) {
        int maxInterval = 28;
        IntVar[] intervalsShifted = makeIntVarArray(intervals.length,
                i -> sum(intervals[i], makeIntVar(cp, maxInterval, maxInterval)));
        int[] skipMotion = new int[57];
        for (int i = -maxInterval; i <= maxInterval; i++)
            skipMotion[maxInterval + i] = 1;

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

        int half = (n - 1)/2 - 1;
        int remainingAllowedSkips = half - nSkips;
        return sum(sum(skips), makeIntVar(cp, -remainingAllowedSkips, -remainingAllowedSkips));
    }

    /**
     * Returns a IntVar indicating the number of intervals violating the characteristicModalSkips constraint
     * We want a minimum of three intervals between C and G or G and C
     * C: 5, 17
     * G: 12
     *
     * @param pitches array indicating the future notes
     * @param intervals array indication the future intervals
     * @param previousNotes array indicating the previous notes
     * @return a IntVar indicating the number of violations
     */
    private static IntVar characteristicModalSkips(IntVar[] pitches, IntVar[] intervals, ArrayList<Integer> previousNotes) {
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

        return sum(
                makeIntVar(cp, minNbCharacteristicModalSkips, minNbCharacteristicModalSkips),
                minus(sum(isSkip))
        );

    }

    /**
     * Returns a IntVar indicating the number of pitches violating the coverModalRange constraint
     * We want the notes in the range {5, 7, 9, 10, 12, 14, 16, 17, 19} to occur at least once
     *
     * @param pitches array indicating the future notes
     * @param previousNotes array indicating the previous notes
     * @return a IntVar indicating the number of violations
     */
    private static IntVar coverModalRange(IntVar[] pitches, ArrayList<Integer> previousNotes) {
        int[] pitchesToRestrict = new int[]{0, 2, 5, 7, 9, 10, 12, 14, 16, 17, 19, 21, 22, 24, 26, 27, 28};
        IntVar[] pitchOccurrences = makeIntVarArray(cp, pitchesToRestrict.length, 0, n);
        IntVar[] pitchesFuture = makeIntVarArray(previousNotes.size() > 0 ? pitches.length - 1 : pitches.length, i -> previousNotes.size() > 0 ? pitches[i + 1] : pitches[i]);
        cp.post(cardinality(pitchesFuture, pitchesToRestrict, pitchOccurrences));

        HashMap<Integer, Integer> pastOccurrences = new HashMap<>();
        for (int pitch : pitchesToRestrict)
            pastOccurrences.put(pitch, 0);

        for (Integer note : previousNotes) {
            if (pastOccurrences.containsKey(note)) {
                pastOccurrences.put(note, pastOccurrences.get(note) + 1);
            }
        }


        IntVar[] violations = makeIntVarArray(cp, pitchesToRestrict.length, 0, pitches.length);
        for (int i = 0; i < pitchesToRestrict.length; i++) {
            if (pitchesToRestrict[i] >= 5 && pitchesToRestrict[i] <= 19) {
                if (pastOccurrences.get(pitchesToRestrict[i]) == 0) {
                    violations[i] = isLess(pitchOccurrences[i], 1);
                } else {
                    violations[i].assign(0);
                }
            } else {
                violations[i] = pitchOccurrences[i];
            }
        }
        return sum(violations);
    }

    /**
     * Returns a IntVar indicating the number of intervals violating the noRepeat constraint
     * We forbid intervals of 0
     *
     * @param intervalsFuture array indicating the future intervals
     * @return a IntVar indicating the number of violations
     */
    private static IntVar noRepeat(IntVar[] intervalsFuture) {
        IntVar[] repeats = makeIntVarArray(cp, 1, 0, intervalsFuture.length);
        cp.post(cardinality(intervalsFuture, new int[]{0}, repeats));
        return repeats[0];
    }

    /**
     * Returns a BoolVar indicating if the stepwiseDescentToFinal constraint is violated
     * We want the last interval to be either -2 or -1
     *
     * @param intervals array indicating the future intervals
     * @return a BoolVar indicating if there is a violation
     */
    private static BoolVar stepwiseDescentToFinal(IntVar[] intervals) {
        return isLarger(sum(isLess(intervals[intervals.length - 1], -2), isLarger(intervals[intervals.length - 1], -1)), 0);
    }

    /**
     * Returns a BoolVar indicating if the tonicEnds constraint is violated
     * We want the last note to be the tonic
     *
     * @param pitches array indicating the future notes
     * @return a BoolVar indicating if there is a violation
     */
    private static BoolVar tonicEnds(IntVar[] pitches) {
        return isLarger(sum(isLess(pitches[pitches.length - 1], tonic), isLarger(pitches[pitches.length - 1], tonic)), 0);
    }

    /**
     * Returns a IntVar indicating the number of intervals violating the tritonOutlines constraint
     * An outline of an augmented fourth is prohibited.
     * An outline of a diminished fifth is allowed only if it is completely filled in by step (interval smaller than
     * 2) and then followed by a step in the opposite direction.
     *
     * @param intervalsShiftedFuture array indicating the future intervals (shifted between 0 and 24)
     * @param previousNotes array indicating the previous notes
     * @return a IntVar indicating the number of violations
     */
    private static IntVar tritonOutlines(IntVar[] intervalsShiftedFuture, ArrayList<Integer> previousNotes) {
        int nbStates = 1 + 2 * 13 + 1;
        int error = 27;
        int beginningState = 0;
        ArrayList<Integer> goalStates = (ArrayList<Integer>) IntStream.range(0, nbStates).boxed().collect(Collectors.toList());

        
        int[][] automaton = new int[nbStates][12 - (-12) + 1];

        //If the note is repeated, we do not change the state
        for( int i=0; i<automaton.length; i++)
            automaton[i][12+0] = i;

        // INITIAL STATE
        for(int j=1; j<=5; j++) {
            automaton[0][12+j] = j; // start ascending outline
            automaton[0][12-j] = 13+j; // start descending outline
        }
        for(int j = 6 ; j<=12; j++) {
            automaton[0][12+j] = 13; // ascending outline beyond tritone
            automaton[0][12-j] = 13+13; // descending outline beyond tritone
        }
        // 2-NOTES STATES
        // states 1 and 13+1
        automaton[1][12+1] = error; // no transition on two consecutive semi-tones
        automaton[13+1][12-1] = error; // no transition on two consecutive semi-tones
        for(int j=2; j<=5; j++) { // continue with outline
            automaton[1][12+j] = 4+j;
            automaton[13+1][12-j] = 13+4+j;
        }
        for(int j = 6 ; j<=12; j++) { // outline beyond tritone
            automaton[1][12+j] = 13;
            automaton[13+1][12-j] = 13+13;
        }
        for(int j=1; j<=5; j++) {
            automaton[13+1][12+j] = j; // start ascending outline
            automaton[1][12-j] = 13+j; // start descending outline
        }
        for(int j=7; j<=12; j++) {
            automaton[13+1][12+j] = 13; // ascending outline beyond tritone
            automaton[1][12-j] = 13+13; // descending outline beyond tritone
        }
        // other 2-notes states
        for(int i=2; i<=5; i++) {
            for(int j=1; j<=6-i; j++) { // continue with outline
                automaton[i][12+j] = i+3+j;
                automaton[13+i][12-j] = 13+i+3+j;
            }
            for(int j = 7 ; j<=12; j++) { // outline beyond tritone
                automaton[i][12+j] = 13;
                automaton[13+i][12-j] = 13+13;
            }
            for(int j=1; j<=5; j++) {
                automaton[13+i][12+j] = j; // start ascending outline
                automaton[i][12-j] = 13+j; // start descending outline
            }
            for(int j=7; j<=12; j++) {
                automaton[13+i][12+j] = 13; // ascending outline beyond tritone
                automaton[i][12-j] = 13+13; // descending outline beyond tritone
            }
        }
        // 3-NOTES STATES
        // states 6 and 13+6
        automaton[6][12+1] = error; // no transition on too close semi-tones
        automaton[13+6][12-1] = error; // no transition on too close semi-tones
        for(int j=2; j<=3; j++) { // continue with outline
            automaton[6][12+j] = 8+j;
            automaton[13+6][12-j] = 13+8+j;
        }
        for(int j = 4 ; j<=12; j++) { // outline beyond tritone
            automaton[6][12+j] = 13;
            automaton[13+6][12-j] = 13+13;
        }
        for(int j=1; j<=5; j++) {
            automaton[13+6][12+j] = j; // start ascending outline
            automaton[6][12-j] = 13+j; // start descending outline
        }
        for(int j=7; j<=12; j++) {
            automaton[13+6][12+j] = 13; // ascending outline beyond tritone
            automaton[6][12-j] = 13+13; // descending outline beyond tritone
        }
        // other 3-notes states
        for(int i=7; i<=9; i++) {
            for(int j=1; j<=9-i; j++) { // continue with outline
                automaton[i][12+j] = i+2+j;
                automaton[13+i][12-j] = 13+i+2+j;
            }
            for(int j = 10 ; j<=12; j++) { // outline beyond tritone
                automaton[i][12+j] = 13;
                automaton[13+i][12-j] = 13+13;
            }
            for(int j=1; j<=5; j++) {
                automaton[13+i][12+j] = j; // start ascending outline
                automaton[i][12-j] = 13+j; // start descending outline
            }
            for(int j=7; j<=12; j++) {
                automaton[13+i][12+j] = 13; // ascending outline beyond tritone
                automaton[i][12-j] = 13+13; // descending outline beyond tritone
            }
        }
        // but actually states 9 and 13+9 cannot end an outline so..
        for(int j=1; j<=12; j++) {
            automaton[13+9][12+j] = error;
            automaton[9][12-j] = error;
        }
        // 4-NOTES STATES
        // states 10 and 13+10
        automaton[10][12+1] = 12; // continue with outline
        automaton[13+10][12-1] = 13+12;
        for(int j=2; j<=12; j++) { // outline beyond tritone
            automaton[10][12+j] = 13;
            automaton[13+10][12-j] = 13+13;
        }
        for(int j=1; j<=5; j++) {
            automaton[13+10][12+j] = j; // start ascending outline
            automaton[10][12-j] = 13+j; // start descending outline
        }
        for(int j=7; j<=12; j++) {
            automaton[13+10][12+j] = 13; // ascending outline beyond tritone
            automaton[10][12-j] = 13+13; // descending outline beyond tritone
        }
        // states 11 and 13+11
        for(int j=1; j<=12; j++) { // outline beyond tritone
            automaton[11][12+j] = 13;
            automaton[13+11][12-j] = 13+13;
        }
        for(int j=1; j<=12; j++) {
            automaton[13+11][12+j] = error;
            automaton[11][12-j] = error;
        }
        // 5-NOTES STATES
        // states 12 and 13+12
        for(int j=1; j<=12; j++) { // outline beyond tritone
            automaton[12][12+j] = 13;
            automaton[13+12][12-j] = 13+13;
        }
        for(int j=1; j<=2; j++) {
            automaton[13+12][12+j] = j;
            automaton[12][12-j] = 13+j;
        }
        for(int j=3; j<=12; j++) {
            automaton[13+12][12+j] = error;
            automaton[12][12-j] = error;
        }
        // "BEYOND" STATES
        for(int j=1; j<=12; j++) { // outline beyond tritone
            automaton[13][12+j] = 13;
            automaton[13+13][12-j] = 13+13;
        }
        for(int j=1; j<=5; j++) {
            automaton[13+13][12+j] = j; // start ascending outline
            automaton[13][12-j] = 13+j; // start descending outline
        }
        for(int j=7; j<=12; j++) {
            automaton[13+13][12+j] = 13; // ascending outline beyond tritone
            automaton[13][12-j] = 13+13; // descending outline beyond tritone
        }

        for (int i = 0; i < 25; i++) {
            automaton[error][i] = error;
        }

        int[][] costs = new int[nbStates][12 - (-12) + 1];
        for (int i = 0; i < nbStates; i++) {
            for (int j = 0; j < costs[i].length; j++) {
                // Only transitions that lead to the error state will have a cost
                if (automaton[i][j] == error && i != error) {
                    costs[i][j] = 1;
                } else {
                    costs[i][j] = 0;
                }
            }
        }

        // Figuring out beginning state:
        int state = beginningState;
        for(int i = 1; i < previousNotes.size(); i++) {
            int interval = previousNotes.get(i) - previousNotes.get(i - 1);
            int intervalShifted = interval + 12;
            // If the interval is not valid, we clip it
            intervalShifted = Math.min(Math.max(intervalShifted, 0), 24);
            // We go through the automaton as usual
            state = automaton[state][intervalShifted];
            // When we reach an error state, instead of being stuck there, we reset
            if (state == error)
                state = beginningState;
        }
        beginningState = state;

        BoolVar cost = makeBoolVar(cp);
        cp.post(costRegular(intervalsShiftedFuture, automaton, beginningState, goalStates, costs, cost));
        return cost;
    }

    /**
     * Propagates the constraints and computes the marginals
     */
    private static void solve() {
        cp.fixPoint();
        cp.beliefPropa();
    }

    /**
     * Assigns the two last chosen notes to their value
     *
     * @param pitches array indicating the relevant notes
     * @param previousNotes array indicating the previous notes
     * @param presentNote the current note chosen
     */
    private static void initializePitches(IntVar[] pitches, ArrayList<Integer> previousNotes, int presentNote) {
        // We set all the previous notes (not the current) to the chosen values
        if (previousNotes.size() > 0) {
            pitches[0].assign(previousNotes.get(previousNotes.size() - 1));
            pitches[1].assign(presentNote);
        } else {
            pitches[0].assign(presentNote);
        }
    }

}
