/*
 *    EDIST2.java
 * 
 *    @author Jorge Chamby-Diaz (jchambyd at gmail dot com)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */
package moa.classifiers;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import moa.AbstractMOAObject;
import moa.core.Measurement;
import moa.options.ClassOption;

/**
 * EDIST2: Error Distance Approach for Drift Detection and Monitoring
 *
 * <p>
 * Error Distance Approach for Drift Detection and Monitoring (EDIST2). EDIST2
 * was proposed in order to deal with these complex drifts EDIST2 monitors the
 * learner performance through a self-adaptive window that is autonomously
 * adjusted through a statistical hypothesis test. This statistical test
 * provides theoretical guarantees, regarding the false alarm rate, which were
 * experimentally confirmed.
 * </p>
 *
 * <p>
 * See details in:<br>
 * Imen Khamassi, Moamar Sayed-Mouchaweh, Moez Hammami Self-Adaptive Windowing
 * Approach for Handling Complex Concept Drift. In Cognitive Computation, DOI:
 * 10.1007/s12559-015-9341-0, Springer, 2015.
 * </p>
 *
 * <p>
 * Parameters:
 * </p>
 * <ul>
 * <li>-l : Base Classiﬁer to train.</li>
 * <li>-n : Maximum number of error occurred in the window</li>
 * </ul>
 *
 * @author Jorge Chamby-Diaz (jchambyd at gmail dot com)
 * @version $Revision: 1 $
 */
public class EDIST2 extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class,
            "trees.HoeffdingTree");

    public IntOption numberErrorsOption = new IntOption("numberErrors", 'n',
            "Maximum number of error occurred in the window.", 30, 1, Integer.MAX_VALUE);

    protected Classifier classifier, bkgClassifier;
    WindowEDIST2 global, current, globalTmp;
    long index;

    @Override
    public void resetLearningImpl() {
        this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
        this.bkgClassifier = this.classifier.copy();
        this.classifier.resetLearning();
        this.bkgClassifier.resetLearning();
        this.global = null;
        this.current = null;
        this.globalTmp = null;
        this.index = 0;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        this.index++;

        if (this.global == null) {
            this.global = new WindowEDIST2(inst, this.numberErrorsOption.getValue());
        }

        if (!this.global.isCompleted()) {
            this.global.buffer.add(inst);
            if (!this.classifier.correctlyClassifies(inst)) {
                this.global.updateParameters(this.index);
            }
        } else {
            if (this.current == null) {
                this.current = new WindowEDIST2(inst, this.numberErrorsOption.getValue());
                this.current.lastError = this.index - 1;
            }

            if (!this.current.isCompleted()) {
                this.current.buffer.add(inst);
                if (!this.classifier.correctlyClassifies(inst)) {
                    this.current.updateParameters(this.index);
                }
            } else {
                int level = this.detectedLevel(global, current);

                switch (level) {
                case 0: // In control
                    this.global.addInstances(this.current.buffer);
                    this.updateParatemeters(global, current);
                    break;
                case 1: // Warning
                    if (this.globalTmp == null) {
                        this.globalTmp = (WindowEDIST2) this.current.copy();
                    } else {
                        this.globalTmp.addInstances(this.current.buffer);
                        this.updateParatemeters(globalTmp, current);
                    }
                    this.bkgClassifier.trainOnInstance(inst);
                    break;
                case 2: // Drift
                    if (this.globalTmp != null)
                        this.global = (WindowEDIST2) this.globalTmp.copy();
                    else
                        this.global = (WindowEDIST2) this.current.copy();
                    this.globalTmp = null;
                    this.classifier = null;
                    this.classifier = this.bkgClassifier;
                    this.bkgClassifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
                    this.bkgClassifier.resetLearning();
                    break;
                default:
                    break;
                }
                // Reset current window
                long tmp = this.current.lastError;
                this.current = new WindowEDIST2(inst, this.numberErrorsOption.getValue());
                this.current.lastError = tmp;
            }
        }

        this.classifier.trainOnInstance(inst);
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return this.classifier.getVotesForInstance(inst);
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        Measurement[] measurements = null;
        return measurements;
    }

    private int detectedLevel(WindowEDIST2 global, WindowEDIST2 current) {
        int level = 0;
        double mu_d = global.mu - current.mu;
        double delta_d = Math.sqrt(Math.pow(global.delta, 2) / global.N + Math.pow(current.delta, 2) / current.N);
        double epsilon = 1.65 * delta_d;

        // Drift
        if (mu_d > epsilon + delta_d)
            level = 2;
        // Warning
        else if (mu_d > epsilon)
            level = 1;

        return level;
    }

    private void updateParatemeters(WindowEDIST2 tglobal, WindowEDIST2 tcurrent) {
        tglobal.mu = (1 / (tglobal.N + tcurrent.N)) * (tglobal.N * tglobal.mu + tcurrent.N + tcurrent.mu);

        tglobal.delta = Math
                .sqrt((((tglobal.N * Math.pow(tglobal.delta, 2)) + (tcurrent.N * Math.pow(tcurrent.delta, 2)))
                        / (tglobal.N + tcurrent.N))
                        + (((tglobal.N * tcurrent.N) / Math.pow(tglobal.N + tcurrent.N, 2))
                                * (Math.pow(tglobal.mu - tcurrent.mu, 2))));

        tglobal.N += tcurrent.N;

        tglobal.lastError = tcurrent.lastError;
    }

    /**
     * Inner class that represents a single data window characterized by: error
     * number (N), error distance mean (mu) and error distance standard deviation
     * (delta).
     */
    protected final class WindowEDIST2 extends AbstractMOAObject {
        private static final long serialVersionUID = 1L;
        // Buffer os instances
        protected Instances buffer;
        // Error distance mean
        public double mu;
        // Error distance standard deviation
        public double delta;
        // Number of error in the window
        public int N;
        // Maximum number of error in the window
        public int maxN;
        // Window is completed
        public boolean completed;
        // Index last error
        public long lastError;

        public WindowEDIST2(Instance inst, int maxN) {
            this.mu = 0;
            this.delta = 0;
            this.N = 0;
            this.lastError = 0;
            this.buffer = new Instances(inst.dataset());
            this.maxN = maxN;
        }

        public WindowEDIST2(Instances instances) {
            this.mu = 0;
            this.delta = 0;
            this.N = 0;
            this.lastError = 0;
            this.buffer = instances;
        }

        public boolean isCompleted() {
            return this.completed;
        }

        public void addInstances(Instances instances) {
            for (int i = 0; i < instances.numInstances(); i++) {
                this.buffer.add(instances.get(i));
            }
        }

        public void updateParameters(long indexError) {
            long d = indexError - this.lastError;

            this.mu = (this.N / (double) (this.N + 1)) * this.mu + d / (double) (this.N + 1);

            this.delta = Math.sqrt((N - 1) * Math.pow(this.delta, 2) / (this.N + Double.MIN_VALUE)
                    + Math.pow(d - this.mu, 2) / (double) (this.N + 1));
            this.N += 1;

            this.lastError = indexError;

            if (this.N == this.maxN)
                this.completed = true;
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {
        }
    }
}
