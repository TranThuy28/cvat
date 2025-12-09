// Copyright (C) 2020-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { ReactPortal } from 'react';
import ReactDOM from 'react-dom';
import { connect } from 'react-redux';
import Icon, {
    EnvironmentFilled,
    EnvironmentOutlined,
    LoadingOutlined,
    QuestionCircleOutlined,
} from '@ant-design/icons';
import Popover from 'antd/lib/popover';
import Select from 'antd/lib/select';
import Button from 'antd/lib/button';
import InputNumber from 'antd/lib/input-number';
import Modal from 'antd/lib/modal';
import Slider from 'antd/lib/slider';
import Text from 'antd/lib/typography/Text';
import Tabs from 'antd/lib/tabs';
import { Row, Col } from 'antd/lib/grid';
import notification from 'antd/lib/notification';
import message from 'antd/lib/message';
import Switch from 'antd/lib/switch';
import lodash, { omit } from 'lodash';

import { AIToolsIcon } from 'icons';
import { Canvas, convertShapesForInteractor } from 'cvat-canvas-wrapper';
import {
    getCore, Label, MLModel, ObjectState, ObjectType, ShapeType, Job,
    MinimalShape, InteractorResults, TrackerResults,
} from 'cvat-core-wrapper';
import openCVWrapper, { MatType } from 'utils/opencv-wrapper/opencv-wrapper';
import { clamp } from 'utils/math';
import {
    CombinedState, ActiveControl, ToolsBlockerState,
} from 'reducers';
import {
    interactWithCanvas,
    switchNavigationBlocked as switchNavigationBlockedAction,
    fetchAnnotationsAsync,
    updateAnnotationsAsync,
    createAnnotationsAsync,
} from 'actions/annotation-actions';
import DetectorRunner, { AnnotateTaskRequestBody } from 'components/model-runner-modal/detector-runner';
import LabelSelector from 'components/label-selector/label-selector';
import CVATTooltip from 'components/common/cvat-tooltip';
import CVATMarkdown from 'components/common/cvat-markdown';

import ApproximationAccuracy, {
    thresholdFromAccuracy,
} from 'components/annotation-page/standard-workspace/controls-side-bar/approximation-accuracy';
import { switchToolsBlockerState } from 'actions/settings-actions';
import withVisibilityHandling from './handle-popover-visibility';
import ToolsTooltips from './interactor-tooltips';

interface StateToProps {
    canvasInstance: Canvas;
    labels: Label[];
    states: ObjectState[];
    activeLabelID: number | null;
    jobInstance: Job;
    isActivated: boolean;
    frame: number;
    interactors: MLModel[];
    detectors: MLModel[];
    trackers: MLModel[];
    curZOrder: number;
    defaultApproxPolyAccuracy: number;
    toolsBlockerState: ToolsBlockerState;
    frameIsDeleted: boolean;
}

interface DispatchToProps {
    updateAnnotations: (states: ObjectState[]) => Promise<void>;
    createAnnotations: (states: ObjectState[]) => Promise<void>;
    fetchAnnotations: () => Promise<void>;
    onInteractionStart: typeof interactWithCanvas;
    onSwitchToolsBlockerState: typeof switchToolsBlockerState;
    switchNavigationBlocked: typeof switchNavigationBlockedAction;
}

const MIN_SUPPORTED_INTERACTOR_VERSION = 2;
const core = getCore();
const CustomPopover = withVisibilityHandling(Popover, 'tools-control');

function mapStateToProps(state: CombinedState): StateToProps {
    const {
        annotation: {
            job: { instance: jobInstance, labels },
            canvas: { instance: canvasInstance, activeControl },
            player: {
                frame: { number: frame, data: { deleted: frameIsDeleted } },
            },
            annotations: {
                zLayer: { cur: curZOrder },
                states,
            },
            drawing: { activeLabelID },
        },
        models: {
            interactors, detectors, trackers,
        },
        settings: {
            workspace: { toolsBlockerState, defaultApproxPolyAccuracy },
        },
    } = state;

    return {
        interactors,
        detectors,
        trackers,
        isActivated: activeControl === ActiveControl.AI_TOOLS,
        activeLabelID,
        labels,
        states,
        canvasInstance: canvasInstance as Canvas,
        jobInstance: jobInstance as Job,
        frame,
        curZOrder,
        defaultApproxPolyAccuracy,
        toolsBlockerState,
        frameIsDeleted,
    };
}

const mapDispatchToProps = {
    onInteractionStart: interactWithCanvas,
    updateAnnotations: updateAnnotationsAsync,
    createAnnotations: createAnnotationsAsync,
    fetchAnnotations: fetchAnnotationsAsync,
    onSwitchToolsBlockerState: switchToolsBlockerState,
    switchNavigationBlocked: switchNavigationBlockedAction,
};

type Props = StateToProps & DispatchToProps;
interface TrackedShape {
    clientID: number;
    serverlessState: any;
    shapePoints: number[];
    trackerModel: MLModel;
}

interface State {
    activeInteractor: MLModel | null;
    activeLabelID: number | null;
    activeTracker: MLModel | null;
    startInteractingWithBox: boolean;
    convertMasksToPolygons: boolean;
    trackedShapes: TrackedShape[];
    fetching: boolean;
    pointsReceived: boolean;
    approxPolyAccuracy: number;
    mode: 'detection' | 'interaction' | 'tracking';
    portals: React.ReactPortal[];
    maskThreshold: number | null;
    logitLoading: boolean;
}

interface LogitCacheEntry {
    map: number[][];
    width: number;
    height: number;
}

type DetectorResults = Extract<Awaited<ReturnType<typeof core.lambda.call>>, { version: number }>;

function trackedRectangleMapper(shape: MinimalShape): MinimalShape {
    return {
        type: ShapeType.RECTANGLE,
        points: shape.points.reduce(
            (acc: number[], value: number, index: number): number[] => {
                if (index % 2) {
                // y
                    acc[1] = Math.min(acc[1], value);
                    acc[3] = Math.max(acc[3], value);
                } else {
                // x
                    acc[0] = Math.min(acc[0], value);
                    acc[2] = Math.max(acc[2], value);
                }
                return acc;
            },
            [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER, Number.MIN_SAFE_INTEGER, Number.MIN_SAFE_INTEGER],
        ),
    };
}

function registerPlugin(): (callback: null | (() => void)) => void {
    let onTrigger: null | (() => void) = null;
    const listener = {
        name: 'Remove annotations listener',
        description: 'Tracker needs to know when annotations is reset in the job',
        cvat: {
            classes: {
                Job: {
                    prototype: {
                        annotations: {
                            clear: {
                                leave(self: any, result: any) {
                                    if (typeof onTrigger === 'function') {
                                        onTrigger();
                                    }
                                    return result;
                                },
                            },
                        },
                    },
                },
            },
        },
    };

    core.plugins.register(listener);

    return (callback: null | (() => void)) => {
        onTrigger = callback;
    };
}

const onRemoveAnnotations = registerPlugin();

export class ToolsControlComponent extends React.PureComponent<Props, State> {
    private interaction: {
        id: string | null;
        isAborted: boolean;
        latestResponse: {
            rle: number[];
            points: [number, number][];
            bounds?: [number, number, number, number];
            mask?: number[][];
            threshold: number | null;
        };
        latestPostponedEvent: Event | null;
        latestApproximatedPoints: number[][];
        latestRequest: null | {
            interactor: MLModel;
            data: {
                frame: number;
                neg_points: number[][];
                pos_points: number[][];
                obj_bbox: number[][];
            };
        } | null;
        hideMessage: (() => void) | null;
    };
    private maskPreviewUpdateInProgress = false;
    private pendingMaskPreview: number | null = null;
    private logitCache = new Map<number, LogitCacheEntry>();
    private logitLoadSeq = 0;
    private logitAbortController: AbortController | null = null;
    private isComponentUnmounted = false;
    private sliderDebounceTimer: number | null = null;
    private lastRenderedThreshold: number | null = null;
    private rleCache = new Map<number, number[]>(); // Cache RLE by threshold (rounded to 2 decimals)

    public constructor(props: Props) {
        super(props);

        const supportedTrackers = this.getSupportedTrackers();

        this.state = {
            convertMasksToPolygons: false,
            startInteractingWithBox: false,
            activeInteractor: props.interactors.length ? props.interactors[0] : null,
            activeTracker: supportedTrackers.length ? supportedTrackers[0] : null,
            activeLabelID: props.labels.length ? props.labels[0].id as number : null,
            approxPolyAccuracy: props.defaultApproxPolyAccuracy,
            trackedShapes: [],
            fetching: false,
            pointsReceived: false,
            mode: 'interaction',
            portals: [],
            maskThreshold: null,
            logitLoading: false,
        };

        this.interaction = {
            id: null,
            isAborted: false,
            latestPostponedEvent: null,
            latestResponse: {
                rle: [],
                points: [],
                mask: [],
                threshold: null,
            },
            latestApproximatedPoints: [],
            latestRequest: null,
            hideMessage: null,
        };
    }

    public componentDidMount(): void {
        const { canvasInstance } = this.props;
        onRemoveAnnotations(() => {
            this.setState({ trackedShapes: [] });
        });

        this.setState({
            portals: this.collectTrackerPortals(),
        });

        canvasInstance.html().addEventListener('canvas.interacted', this.interactionListener);
        canvasInstance.html().addEventListener('canvas.canceled', this.cancelListener);
    }

    public componentDidUpdate(prevProps: Props, prevState: State): void {
        const {
            isActivated, defaultApproxPolyAccuracy, canvasInstance, states, toolsBlockerState,
            frame, jobInstance,
        } = this.props;
        const {
            approxPolyAccuracy,
            mode,
            activeTracker,
            convertMasksToPolygons,
            maskThreshold,
        } = this.state;

        if (prevProps.jobInstance !== jobInstance) {
            this.logitCache.clear();
        }

        if (prevProps.states !== states || prevState.activeTracker !== activeTracker) {
            this.setState({
                portals: this.collectTrackerPortals(),
            });
        }

        if (prevProps.isActivated && !isActivated) {
            window.removeEventListener('contextmenu', this.contextmenuDisabler);
            this.logitAbortController?.abort();
            // hide interaction message if exists
            if (this.interaction.hideMessage) {
                this.interaction.hideMessage();
                this.interaction.hideMessage = null;
            }
        } else if (!prevProps.isActivated && isActivated) {
            // reset flags when start interaction/tracking
            this.interaction = {
                id: null,
                isAborted: false,
                latestPostponedEvent: null,
                latestResponse: {
                    rle: [],
                    points: [],
                    mask: [],
                    threshold: null,
                },
                latestApproximatedPoints: [],
                latestRequest: null,
                hideMessage: null,
            };
            this.pendingMaskPreview = null;
            this.maskPreviewUpdateInProgress = false;

            this.setState({
                approxPolyAccuracy: defaultApproxPolyAccuracy,
                pointsReceived: false,
                maskThreshold: null,
            });
            window.addEventListener('contextmenu', this.contextmenuDisabler);
        }

        if (
            prevProps.toolsBlockerState.algorithmsLocked &&
            !toolsBlockerState.algorithmsLocked &&
            isActivated && mode === 'interaction' && this.interaction.latestPostponedEvent
        ) {
            this.onInteraction(this.interaction.latestPostponedEvent);
        }

        if (prevState.approxPolyAccuracy !== approxPolyAccuracy) {
            if (isActivated && mode === 'interaction' && this.interaction.latestResponse.points.length) {
                this.approximateResponsePoints(this.interaction.latestResponse.points)
                    .then((points: number[][]) => {
                        this.interaction.latestApproximatedPoints = points;
                        canvasInstance.interact({
                            enabled: true,
                            shapeType: ShapeType.POLYGON,
                            intermediateShape: {
                                shapeType: ShapeType.POLYGON,
                                points: this.interaction.latestApproximatedPoints.flat(),
                            },
                        });
                    });
            }
        }

        if (
            prevState.convertMasksToPolygons !== convertMasksToPolygons &&
            maskThreshold !== null &&
            this.interaction.latestResponse.mask?.length &&
            mode === 'interaction' // ← CHỈ CHECK MODE
        ) {
            void this.updateMaskPreview(maskThreshold);
        }

        if (
            prevProps.frame !== frame &&
            mode === 'interaction' && // ← CHỈ CHECK MODE
            this.interaction.id
        ) {
            void this.loadLogitForFrame(frame);
        }

        this.checkTrackedStates(prevProps);
    }

    public componentWillUnmount(): void {
        const { canvasInstance } = this.props;
        this.isComponentUnmounted = true;
        onRemoveAnnotations(null);
        canvasInstance.html().removeEventListener('canvas.interacted', this.interactionListener);
        canvasInstance.html().removeEventListener('canvas.canceled', this.cancelListener);
        this.logitAbortController?.abort();
        if (this.sliderDebounceTimer !== null) {
            window.clearTimeout(this.sliderDebounceTimer);
            this.sliderDebounceTimer = null;
        }
    }

    private getSupportedTrackers(): MLModel[] {
        const { trackers } = this.props;
        return trackers.filter((tracker: MLModel) => tracker.supportedShapeTypes!.includes(ShapeType.RECTANGLE));
    }

    private contextmenuDisabler = (e: MouseEvent): void => {
        if (
            e.target &&
            (e.target as Element).classList &&
            (e.target as Element).classList.toString().includes('ant-modal')
        ) {
            e.preventDefault();
        }
    };

    private cancelListener = async (): Promise<void> => {
        const { fetching } = this.state;
        if (fetching) {
            // user pressed ESC
            this.setState({ fetching: false });
            this.interaction.isAborted = true;
        }
    };

    private runInteractionRequest = async (interactionId: string): Promise<void> => {
        const { jobInstance, canvasInstance } = this.props;
        const { activeInteractor, fetching } = this.state;

        const { id, latestRequest } = this.interaction;
        if (id !== interactionId || !latestRequest || fetching) {
            // current interaction request is not relevant (new interaction session has started)
            // or a user didn't add more points
            // or one server request is on processing
            return;
        }

        const { interactor, data } = latestRequest;
        this.interaction.latestRequest = null;

        try {
            this.interaction.hideMessage = message.loading({
                content: `Waiting for a response from ${activeInteractor?.name}`,
                duration: 0,
                className: 'cvat-tracking-notice',
            });
            try {
                // run server request
                this.setState({ fetching: true });

                const response = await core.lambda.call(
                    jobInstance.taskId,
                    interactor,
                    { ...data, job: jobInstance.id },
                ) as InteractorResults;

                // if only mask presented, let's receive points
                if (response.mask && !response.points) {
                    const left = response.bounds ? response.bounds[0] : 0;
                    const top = response.bounds ? response.bounds[1] : 0;
                    response.points = await this.receivePointsFromMask(response.mask, left, top);
                }

                const hasMask = Array.isArray(response.mask) && response.mask.length > 0;
                let approximated: number[][] = [];

                if (response.points?.length) {
                    approximated = await this.approximateResponsePoints(response.points as [number, number][]);
                }

                if (this.interaction.id !== interactionId || this.interaction.isAborted) {
                    // new interaction session or the session is aborted
                    return;
                }

                if (hasMask) {
                    const defaultThreshold = this.normalizeThreshold((response as { threshold?: number }).threshold);

                    this.interaction.latestResponse = {
                        bounds: response.bounds,
                        points: (response.points as [number, number][] | undefined) ?? [],
                        rle: [],
                        mask: response.mask as number[][],
                        threshold: defaultThreshold,
                    };

                    if (!approximated.length) {
                        const [left, top] = response.bounds ? [response.bounds[0], response.bounds[1]] : [0, 0];
                        const fallbackPoints = await this.receivePointsFromMask(
                            response.mask as number[][],
                            left,
                            top,
                        );
                        approximated = await this.approximateResponsePoints(fallbackPoints);
                    }

                    this.interaction.latestApproximatedPoints = approximated;
                    this.setState({
                        maskThreshold: defaultThreshold,
                        pointsReceived: !!response.points?.length || hasMask,
                    });

                    await this.updateMaskPreview(defaultThreshold);
                } else {
                    this.interaction.latestResponse = {
                        bounds: response.bounds,
                        points: (response.points as [number, number][] | undefined) ?? [],
                        rle: [],
                        mask: [],
                        threshold: null,
                    };
                    this.interaction.latestApproximatedPoints = approximated;

                    this.setState({
                        pointsReceived: !!response.points?.length,
                        maskThreshold: null,
                    });

                    if (this.interaction.latestApproximatedPoints.length) {
                        canvasInstance.interact({
                            enabled: true,
                            shapeType: ShapeType.POLYGON,
                            intermediateShape: {
                                shapeType: ShapeType.POLYGON,
                                points: this.interaction.latestApproximatedPoints.flat(),
                            },
                        });
                    }
                }
            } finally {
                if (this.interaction.id === interactionId && this.interaction.hideMessage) {
                    this.interaction.hideMessage();
                    this.interaction.hideMessage = null;
                }

                this.setState({ fetching: false });
            }

            setTimeout(() => this.runInteractionRequest(interactionId));
        } catch (error: any) {
            notification.error({
                description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                message: 'Interaction error occurred',
                duration: null,
            });
        }
    };

    private onInteraction = (e: Event): void => {
        const { frame, isActivated } = this.props;
        const { activeInteractor } = this.state;

        if (!isActivated) {
            return;
        }

        if (!this.interaction.id) {
            this.interaction.id = lodash.uniqueId('interaction_');
        }

        const { shapesUpdated, isDone, shapes } = (e as CustomEvent).detail;
        if (isDone) {
            // make an object from current result
            // do not make one more request
            // prevent future requests if possible
            this.interaction.isAborted = true;
            this.interaction.latestRequest = null;
            if (this.interaction.latestApproximatedPoints.length) {
                this.constructFromPoints();
            }
        } else if (shapesUpdated) {
            const interactor = activeInteractor as MLModel;
            this.interaction.latestRequest = {
                interactor,
                data: {
                    frame,
                    obj_bbox: convertShapesForInteractor(shapes, 'rectangle', 0),
                    pos_points: convertShapesForInteractor(shapes, 'points', 0),
                    neg_points: convertShapesForInteractor(shapes, 'points', 2),
                },
            };

            this.runInteractionRequest(this.interaction.id);
        }
    };

    private onTracking = async (e: Event): Promise<void> => {
        const { trackedShapes, activeTracker, activeLabelID } = this.state;
        const {
            isActivated, jobInstance, frame, curZOrder, fetchAnnotations,
        } = this.props;

        if (!isActivated || !activeLabelID) {
            return;
        }

        const [label] = jobInstance.labels.filter((_label: any): boolean => _label.id === activeLabelID);

        const { isDone, shapesUpdated } = (e as CustomEvent).detail;
        if (!isDone || !shapesUpdated) {
            return;
        }

        try {
            const { points } = (e as CustomEvent).detail.shapes[0];
            const state = new core.classes.ObjectState({
                shapeType: ShapeType.RECTANGLE,
                objectType: ObjectType.TRACK,
                source: core.enums.Source.SEMI_AUTO,
                zOrder: curZOrder,
                label,
                points,
                frame,
                occluded: false,
                attributes: {},
                descriptions: [`Trackable (${activeTracker?.name})`],
            });

            const [clientID] = await jobInstance.annotations.put([state]);
            this.setState({
                trackedShapes: [
                    ...trackedShapes,
                    {
                        clientID,
                        serverlessState: null,
                        shapePoints: points,
                        trackerModel: activeTracker as MLModel,
                    },
                ],
            });

            // update annotations on a canvas
            fetchAnnotations();
        } catch (error: any) {
            notification.error({
                description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                message: 'Tracking error occurred',
                duration: null,
            });
        }
    };

    private interactionListener = async (e: Event): Promise<void> => {
        const { toolsBlockerState } = this.props;
        const { mode } = this.state;

        if (mode === 'interaction') {
            if (toolsBlockerState.algorithmsLocked) {
                this.interaction.latestPostponedEvent = e;
                return;
            }

            await this.onInteraction(e);
        }

        if (mode === 'tracking') {
            await this.onTracking(e);
        }
    };

    private setActiveInteractor = (value: string): void => {
        const { interactors } = this.props;
        const [interactor] = interactors.filter((_interactor: MLModel) => _interactor.id === value);

        if (interactor.version < MIN_SUPPORTED_INTERACTOR_VERSION) {
            notification.warning({
                message: 'Interactor API is outdated',
                description: 'Probably, you should consider updating the serverless function',
            });
        }

        this.setState({
            activeInteractor: interactor,
        });
    };

    private setActiveTracker = (value: string): void => {
        const { trackers } = this.props;
        this.setState({
            activeTracker: trackers.filter((tracker: MLModel) => tracker.id === value)[0],
        });
    };

    private collectTrackerPortals(): React.ReactPortal[] {
        const { states, fetchAnnotations } = this.props;
        const { trackedShapes, activeTracker } = this.state;

        const trackedClientIDs = trackedShapes.map((trackedShape: TrackedShape) => trackedShape.clientID);
        const portals = !activeTracker ?
            [] :
            states
                .filter((objectState) => objectState.objectType === 'track' && objectState.shapeType === 'rectangle')
                .map((objectState: any): React.ReactPortal | null => {
                    const { clientID } = objectState;
                    const selectorID = `#cvat-objects-sidebar-state-item-${clientID}`;
                    let targetElement = window.document.querySelector(
                        `${selectorID} .cvat-object-item-button-prev-keyframe`,
                    ) as HTMLElement;

                    const isTracked = trackedClientIDs.includes(clientID);
                    if (targetElement) {
                        targetElement = targetElement.parentElement?.parentElement as HTMLElement;
                        return ReactDOM.createPortal(
                            <Col>
                                {isTracked ? (
                                    <CVATTooltip overlay='Disable tracking'>
                                        <EnvironmentFilled
                                            onClick={() => {
                                                const filteredStates = trackedShapes.filter(
                                                    (trackedShape: TrackedShape) => trackedShape.clientID !== clientID,
                                                );
                                                /* eslint no-param-reassign: ["error", { "props": false }] */
                                                objectState.descriptions = [];
                                                objectState.save().then(() => {
                                                    this.setState({
                                                        trackedShapes: filteredStates,
                                                    });
                                                    fetchAnnotations();
                                                });
                                            }}
                                        />
                                    </CVATTooltip>
                                ) : (
                                    <CVATTooltip overlay={`Enable tracking using ${activeTracker.name}`}>
                                        <EnvironmentOutlined
                                            onClick={() => {
                                                objectState.descriptions = [`Trackable (${activeTracker.name})`];
                                                objectState.keyframe = true;
                                                objectState.save().then(() => {
                                                    this.setState({
                                                        trackedShapes: [
                                                            ...trackedShapes,
                                                            {
                                                                clientID,
                                                                serverlessState: null,
                                                                shapePoints: objectState.points,
                                                                trackerModel: activeTracker,
                                                            },
                                                        ],
                                                    });
                                                    fetchAnnotations();
                                                });
                                            }}
                                        />
                                    </CVATTooltip>
                                )}
                            </Col>,
                            targetElement,
                        );
                    }

                    return null;
                })
                .filter((portal: ReactPortal | null) => portal !== null);

        return portals as React.ReactPortal[];
    }

    private async checkTrackedStates(prevProps: Props): Promise<void> {
        const {
            frame,
            jobInstance,
            states: objectStates,
            trackers,
            fetchAnnotations,
            switchNavigationBlocked,
        } = this.props;
        const { trackedShapes } = this.state;
        let withServerRequest = false;

        type AccumulatorType = {
            // These maps are indexed by tracker ID.
            stateful: Map<string | number, {
                clientIDs: number[];
                states: any[];
                shapes: MinimalShape[];
            }>;
            stateless: Map<string | number, {
                clientIDs: number[];
                shapes: MinimalShape[];
            }>;
        };

        if (prevProps.frame !== frame && trackedShapes.length) {
            // 1. find all trackable objects on the current frame
            // 2. divide them into two groups: with relevant state, without relevant state
            const trackingData = trackedShapes.reduce<AccumulatorType>(
                (acc: AccumulatorType, trackedShape: TrackedShape): AccumulatorType => {
                    const {
                        serverlessState, shapePoints, clientID, trackerModel,
                    } = trackedShape;
                    const clientState = objectStates.find((_state): boolean => _state.clientID === clientID);
                    const keyframes = clientState?.keyframes;

                    if (
                        !clientState || !keyframes ||
                        keyframes?.prev !== frame - 1 ||
                        (typeof keyframes?.last === 'number' && keyframes?.last >= frame)
                    ) {
                        return acc;
                    }

                    if (clientState && !clientState.outside) {
                        const points = clientState.points as number[];
                        withServerRequest = true;
                        const stateIsRelevant =
                            serverlessState !== null &&
                            points.length === shapePoints.length &&
                            points.every((coord: number, i: number) => coord === shapePoints[i]);
                        if (stateIsRelevant) {
                            const container = acc.stateful.get(trackerModel.id) ?? {
                                clientIDs: [],
                                shapes: [],
                                states: [],
                            };
                            container.clientIDs.push(clientID);
                            container.shapes.push({ type: clientState.shapeType, points });
                            container.states.push(serverlessState);
                            acc.stateful.set(trackerModel.id, container);
                        } else {
                            const container = acc.stateless.get(trackerModel.id) ?? {
                                clientIDs: [],
                                shapes: [],
                            };
                            container.clientIDs.push(clientID);
                            container.shapes.push({ type: clientState.shapeType, points });
                            acc.stateless.set(trackerModel.id, container);
                        }
                    }

                    return acc;
                },
                {
                    stateful: new Map(),
                    stateless: new Map(),
                },
            );

            try {
                if (withServerRequest) {
                    switchNavigationBlocked(true);
                }
                // 3. get relevant state for the second group
                for (const [trackerID, trackableObjects] of trackingData.stateless) {
                    let hideMessage = null;
                    try {
                        const [tracker] = trackers.filter((_tracker: MLModel) => _tracker.id === trackerID);
                        if (!tracker) {
                            throw new Error(`Suitable tracker with ID ${trackerID} not found in tracker list`);
                        }

                        const numOfObjects = trackableObjects.clientIDs.length;
                        hideMessage = message.loading({
                            content: `${tracker.name}: states are being initialized for ${numOfObjects} ${
                                numOfObjects > 1 ? 'objects' : 'object'
                            } ..`,
                            duration: 0,
                            className: 'cvat-tracking-notice',
                        });

                        const response = await core.lambda.call(jobInstance.taskId, tracker, {
                            type: 'init_tracking',
                            frame: frame - 1,
                            shapes: trackableObjects.shapes,
                            job: jobInstance.id,
                        }) as TrackerResults;

                        const { states: serverlessStates } = response;
                        const statefulContainer = trackingData.stateful.get(trackerID) ?? {
                            clientIDs: [],
                            shapes: [],
                            states: [],
                        };

                        Array.prototype.push.apply(statefulContainer.clientIDs, trackableObjects.clientIDs);
                        Array.prototype.push.apply(statefulContainer.shapes, trackableObjects.shapes);
                        Array.prototype.push.apply(statefulContainer.states, serverlessStates);
                        trackingData.stateful.set(trackerID, statefulContainer);
                        trackingData.stateless.delete(trackerID);
                    } catch (error: any) {
                        notification.error({
                            message: 'Tracker initialization error',
                            description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                            duration: null,
                        });
                    } finally {
                        if (hideMessage) hideMessage();
                    }
                }

                for (const [trackerID, trackableObjects] of trackingData.stateful) {
                    // 4. run tracking for all the objects
                    let hideMessage = null;
                    try {
                        const [tracker] = trackers.filter((_tracker: MLModel) => _tracker.id === trackerID);
                        if (!tracker) {
                            throw new Error(`Suitable tracker with ID ${trackerID} not found in tracker list`);
                        }

                        const numOfObjects = trackableObjects.clientIDs.length;
                        hideMessage = message.loading({
                            content: `${tracker.name}: ${numOfObjects} ${
                                numOfObjects > 1 ? 'objects are' : 'object is'
                            } being tracked..`,
                            duration: 0,
                            className: 'cvat-tracking-notice',
                        });
                        // eslint-disable-next-line no-await-in-loop
                        const response = await core.lambda.call(jobInstance.taskId, tracker, {
                            type: 'track',
                            frame,
                            states: trackableObjects.states,
                            job: jobInstance.id,
                        }) as TrackerResults;

                        response.shapes = response.shapes.map(trackedRectangleMapper);
                        for (let i = 0; i < trackableObjects.clientIDs.length; i++) {
                            const clientID = trackableObjects.clientIDs[i];
                            const shape = response.shapes[i];
                            const state = response.states[i];
                            const [objectState] = objectStates.filter(
                                (_state: any): boolean => _state.clientID === clientID,
                            );
                            const [trackedShape] = trackedShapes.filter(
                                (_trackedShape: TrackedShape) => _trackedShape.clientID === clientID,
                            );
                            objectState.points = shape.points;
                            objectState.save().then(() => {
                                trackedShape.serverlessState = state;
                                trackedShape.shapePoints = shape.points;
                            });
                        }
                    } catch (error: any) {
                        notification.error({
                            message: 'Tracking error',
                            description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                            duration: null,
                        });
                    } finally {
                        if (hideMessage) hideMessage();
                        fetchAnnotations();
                    }
                }
            } finally {
                if (withServerRequest) {
                    switchNavigationBlocked(false);
                }
            }
        }
    }

    private async constructFromPoints(): Promise<void> {
        const { convertMasksToPolygons } = this.state;
        const {
            frame, labels, curZOrder, activeLabelID, createAnnotations,
        } = this.props;

        if (convertMasksToPolygons) {
            const object = new core.classes.ObjectState({
                frame,
                objectType: ObjectType.SHAPE,
                source: core.enums.Source.SEMI_AUTO,
                label: labels.find((label) => label.id === activeLabelID as number) as Label,
                shapeType: ShapeType.POLYGON,
                points: this.interaction.latestApproximatedPoints.flat(),
                occluded: false,
                zOrder: curZOrder,
            });

            createAnnotations([object]);
        } else {
            const object = new core.classes.ObjectState({
                frame,
                objectType: ObjectType.SHAPE,
                source: core.enums.Source.SEMI_AUTO,
                label: labels.find((label) => label.id === activeLabelID as number) as Label,
                shapeType: ShapeType.MASK,
                points: this.interaction.latestResponse.rle,
                occluded: false,
                zOrder: curZOrder,
            });

            createAnnotations([object]);
        }
    }

    private async initializeOpenCV(): Promise<void> {
        if (!openCVWrapper.isInitialized) {
            const hide = message.loading('OpenCV client initialization..', 0);
            try {
                await openCVWrapper.initialize(() => {});
            } catch (error: any) {
                notification.error({
                    message: 'Could not initialize OpenCV',
                    description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                    duration: null,
                });
            } finally {
                hide();
            }
        }
    }

    private normalizeThreshold(value?: number): number {
        if (typeof value === 'number' && Number.isFinite(value)) {
            return clamp(value, 0, 1);
        }

        return clamp(Math.random(), 0, 1);
    }

    private binarizeMask(mask: number[][], threshold: number): number[][] {
        const normalized = clamp(threshold, 0, 1);
        return mask.map((row: number[]) => row.map((cell: number) => (cell >= normalized ? 1 : 0)));
    }

    private composeRLEFromMask(mask: number[][], bounds?: [number, number, number, number]): number[] {
        if (!mask.length || !mask[0]?.length) {
            return [];
        }

        const rle = core.utils.mask2Rle(mask.flat());
        if (bounds) {
            rle.push(...bounds);
        } else {
            const height = mask.length;
            const width = mask[0].length;
            rle.push(0, 0, width - 1, height - 1);
        }

        return rle;
    }

    private async updateMaskPreview(threshold: number): Promise<void> {
        return this.updateMaskPreviewSmooth(threshold);
    }

    private async updateMaskPreviewSmooth(threshold: number): Promise<void> {
        if (!this.interaction.latestResponse.mask?.length) {
            return;
        }

        if (this.maskPreviewUpdateInProgress) {
            this.pendingMaskPreview = threshold;
            return;
        }

        this.maskPreviewUpdateInProgress = true;

        try {
            if (this.state.mode !== 'interaction') {
                return;
            }

            const normalized = clamp(threshold, 0, 1);
            const currentInteractionId = this.interaction.id;

            if (!currentInteractionId || this.interaction.isAborted) {
                return;
            }

            // Round threshold to 2 decimals for cache key
            const cacheKey = Math.round(normalized * 100) / 100;

            // Check cache first
            let rle = this.rleCache.get(cacheKey);

            if (!rle) {
                // Compute RLE if not cached
                const baseMask = this.interaction.latestResponse.mask as number[][];
                const binarizedMask = this.binarizeMask(baseMask, normalized);
                const bounds = this.interaction.latestResponse.bounds;
                rle = this.composeRLEFromMask(binarizedMask, bounds);

                // Cache the result (limit cache size to prevent memory issues)
                if (this.rleCache.size > 50) {
                    const firstKey = this.rleCache.keys().next().value;
                    if (firstKey !== undefined) {
                        this.rleCache.delete(firstKey);
                    }
                }
                this.rleCache.set(cacheKey, rle);
            }

            if (!rle.length) {
                return;
            }

            this.interaction.latestResponse.rle = rle;
            this.interaction.latestResponse.threshold = normalized;

            const { canvasInstance } = this.props;
            const { convertMasksToPolygons } = this.state;

            // Update immediately without requestAnimationFrame to avoid flickering
            // The canvas will smoothly transition between intermediate shapes
            if (convertMasksToPolygons) {
                const [left, top] = this.interaction.latestResponse.bounds ?
                    [this.interaction.latestResponse.bounds[0], this.interaction.latestResponse.bounds[1]] : [0, 0];

                // For polygons, we still need async processing but do it synchronously
                const binarizedMask = this.binarizeMask(
                    this.interaction.latestResponse.mask as number[][],
                    normalized
                );

                const polygonPoints = await this.receivePointsFromMask(binarizedMask, left, top);
                const approximated = await this.approximateResponsePoints(polygonPoints);

                if (this.interaction.id !== currentInteractionId || this.interaction.isAborted) {
                    return;
                }

                this.interaction.latestApproximatedPoints = approximated;
                this.interaction.latestResponse.points = polygonPoints;

                // Update canvas immediately - canvas handles smooth transition
                canvasInstance.interact({
                    enabled: true,
                    shapeType: ShapeType.POLYGON,
                    intermediateShape: {
                        shapeType: ShapeType.POLYGON,
                        points: approximated.flat(),
                    },
                });

                this.lastRenderedThreshold = normalized;
            } else {
                if (this.interaction.id !== currentInteractionId || this.interaction.isAborted) {
                    return;
                }

                this.interaction.latestResponse.points = [];

                // Update canvas immediately - canvas will smoothly update intermediateShape
                // without clearing the previous mask
                canvasInstance.interact({
                    enabled: true,
                    shapeType: ShapeType.MASK,
                    intermediateShape: {
                        shapeType: ShapeType.MASK,
                        points: rle,
                    },
                });

                this.lastRenderedThreshold = normalized;
            }

        } catch (error) {
            console.error('💥 updateMaskPreview error:', error);
        } finally {
            this.maskPreviewUpdateInProgress = false;
            if (this.pendingMaskPreview !== null) {
                const pendingThreshold = this.pendingMaskPreview;
                this.pendingMaskPreview = null;
                await this.updateMaskPreviewSmooth(pendingThreshold);
            }
        }
    }

    private buildLogitAssetPathCandidates(frame: number): string[] {
        const { jobInstance } = this.props;
        // Backend endpoint: GET /api/jobs/<job_id>/logits/<frame>
        // The backend is responsible for returning the correct image format.
        const base = `/api/jobs/${jobInstance.id}/logits/${frame}`;
        return [base];
    }

    private async fetchLogitAsset(frame: number, signal: AbortSignal): Promise<{ blob: Blob; url: string }> {
        const candidates = this.buildLogitAssetPathCandidates(frame);
        let lastError: Error | null = null;

        for (const url of candidates) {
            try {
                const response = await fetch(url, {
                    signal,
                    cache: 'force-cache',
                    credentials: 'include',
                });
                if (!response.ok) {
                    if (response.status === 404) {
                        console.log(`⚠️ 404 for ${url}, trying next candidate...`);
                        continue;
                    }

                    throw new Error(`Failed to load logit map (${response.status} ${response.statusText})`);
                }

                const blob = await response.blob();
                console.log(`✅ Successfully loaded logit: ${url}`);
                return { blob, url };
            } catch (error: any) {
                if (error?.name === 'AbortError') {
                    throw error;
                }
                lastError = error instanceof Error ? error : new Error(`Failed to load logit map: ${error.message}`);
            }
        }

        throw lastError ?? new Error(`Logit map not found for frame ${frame}. Tried: ${candidates.join(', ')}`);
    }

    private async decodeLogitBlob(blob: Blob): Promise<LogitCacheEntry> {
        let width = 0;
        let height = 0;
        let source: CanvasImageSource | null = null;
        let bitmap: ImageBitmap | null = null;
        try {
            if (typeof createImageBitmap === 'function') {
                bitmap = await createImageBitmap(blob);
                width = bitmap.width;
                height = bitmap.height;
                source = bitmap;
            } else {
                const imageElement = await this.loadImageFromBlob(blob);
                width = imageElement.naturalWidth || imageElement.width;
                height = imageElement.naturalHeight || imageElement.height;
                source = imageElement;
            }

            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            if (!ctx || !source) {
                throw new Error('Unable to create 2D canvas context');
            }

            ctx.drawImage(source, 0, 0);
            const { data } = ctx.getImageData(0, 0, width, height);
            const map: number[][] = new Array(height);
            for (let y = 0; y < height; y++) {
                const row = new Array<number>(width);
                for (let x = 0; x < width; x++) {
                    const idx = (y * width + x) * 4;
                    row[x] = data[idx] / 255;
                }
                map[y] = row;
            }

            return {
                map,
                width,
                height,
            };
        } finally {
            if (bitmap && typeof bitmap.close === 'function') {
                bitmap.close();
            }
        }
    }

    private loadImageFromBlob(blob: Blob): Promise<HTMLImageElement> {
        return new Promise((resolve, reject) => {
            const url = URL.createObjectURL(blob);
            const image = new Image();
            image.onload = () => {
                URL.revokeObjectURL(url);
                resolve(image);
            };
            image.onerror = (event) => {
                URL.revokeObjectURL(url);
                reject(new Error(`Failed to decode logit map (${(event as ErrorEvent)?.message || 'unknown error'})`));
            };
            image.src = url;
        });
    }

    private async requestLogit(frame: number, signal: AbortSignal): Promise<LogitCacheEntry> {
        const { blob } = await this.fetchLogitAsset(frame, signal);
        return this.decodeLogitBlob(blob);
    }

    private async loadLogitForFrame(frame: number): Promise<void> {
        if (!this.interaction.id) {
            this.interaction.id = lodash.uniqueId('interaction_');
        }
        this.interaction.isAborted = false;
        this.pendingMaskPreview = null;
        this.maskPreviewUpdateInProgress = false;
        if (this.isComponentUnmounted) {
            return;
        }

        this.logitAbortController?.abort();
        const abortController = new AbortController();
        this.logitAbortController = abortController;
        const requestSeq = ++this.logitLoadSeq;

        if (!this.isComponentUnmounted) {
            this.setState({
                logitLoading: true,
                pointsReceived: false,
            });
        }

        try {
            const cached = this.logitCache.get(frame);
            const entry = cached ?? await this.requestLogit(frame, abortController.signal);
            if (!cached) {
                this.logitCache.set(frame, entry);
            }

            if (requestSeq !== this.logitLoadSeq) {
                return;
            }

            this.interaction.latestResponse = {
                ...this.interaction.latestResponse,
                rle: [],
                mask: entry.map,
                points: [],
                bounds: [0, 0, entry.width - 1, entry.height - 1],
                threshold: null,
            };
            this.interaction.latestApproximatedPoints = [];
            this.lastRenderedThreshold = null; // Reset when loading new frame
            this.rleCache.clear(); // Clear cache when loading new frame

            const defaultThreshold = 0.5;
            if (!this.isComponentUnmounted) {
                this.setState({
                    maskThreshold: defaultThreshold,
                    pointsReceived: true,
                });
            }

            await this.updateMaskPreview(defaultThreshold);
        } catch (error: any) {
            if (error?.name === 'AbortError') {
                return;
            }

            notification.warning({
                message: 'Logit map unavailable',
                description: <CVATMarkdown>{(error as Error).message}</CVATMarkdown>,
                duration: null,
            });

            this.interaction.latestResponse.mask = [];
            if (!this.isComponentUnmounted) {
                this.setState({
                    maskThreshold: null,
                    pointsReceived: false,
                });
            }
        } finally {
            if (requestSeq === this.logitLoadSeq) {
                if (!this.isComponentUnmounted) {
                    this.setState({ logitLoading: false });
                }
                if (this.logitAbortController === abortController) {
                    this.logitAbortController = null;
                }
            }
        }
    }

    private handleMaskThresholdSliderChange = (value: number): void => {
        const normalized = clamp(value, 0, 1);
        // Update state immediately for responsive UI
        this.setState({ maskThreshold: normalized });

        // Clear previous debounce timer
        if (this.sliderDebounceTimer !== null) {
            window.clearTimeout(this.sliderDebounceTimer);
            this.sliderDebounceTimer = null;
        }

        // Very short debounce (8ms) for smooth updates without flickering
        // This allows updates during drag while preventing excessive renders
        this.sliderDebounceTimer = window.setTimeout(() => {
            this.sliderDebounceTimer = null;
            if (this.interaction.latestResponse.mask?.length && !this.isComponentUnmounted) {
                // Update immediately without checking threshold diff for smoother experience
                void this.updateMaskPreviewSmooth(normalized);
            }
        }, 8); // Very short debounce for responsiveness
    };

    private handleMaskThresholdInputChange = (value: number | string | null): void => {
        if (typeof value !== 'number') {
            return;
        }

        const normalized = clamp(value, 0, 1);
        this.setState({ maskThreshold: normalized });

        if (this.interaction.latestResponse.mask?.length) {
            void this.updateMaskPreview(normalized);
        }
    };

    private async receivePointsFromMask(
        mask: number[][],
        left: number,
        top: number,
    ): Promise<[number, number][]> {
        await this.initializeOpenCV();

        const src = openCVWrapper.mat.fromData(mask[0].length, mask.length, MatType.CV_8UC1, mask.flat());
        try {
            const polygons = openCVWrapper.contours.findContours(src, true);
            return polygons[0].reduce<[number, number][]>((acc, _, idx, array) => {
                if (idx % 2) {
                    acc.push([array[idx - 1] + left, array[idx] + top]);
                }

                return acc;
            }, []);
        } finally {
            src.delete();
        }
    }

    private async approximateResponsePoints(points: number[][]): Promise<number[][]> {
        const { approxPolyAccuracy } = this.state;
        if (points.length > 3) {
            await this.initializeOpenCV();
            const threshold = thresholdFromAccuracy(approxPolyAccuracy);
            return openCVWrapper.contours.approxPoly(points, threshold);
        }

        return points;
    }

    private renderLabelBlock(): JSX.Element {
        const { labels } = this.props;
        const { activeLabelID } = this.state;
        return (
            <>
                <Row justify='start'>
                    <Col>
                        <Text className='cvat-text-color'>Label</Text>
                    </Col>
                </Row>
                <Row justify='center'>
                    <Col span={24}>
                        <LabelSelector
                            style={{ width: '100%' }}
                            labels={labels}
                            value={activeLabelID}
                            onChange={(value: any) => this.setState({ activeLabelID: value.id })}
                        />
                    </Col>
                </Row>
            </>
        );
    }

    private renderTrackerBlock(): JSX.Element {
        const {
            canvasInstance, jobInstance, frame, onInteractionStart,
        } = this.props;
        const { activeTracker, activeLabelID, fetching } = this.state;

        const supportedTrackers = this.getSupportedTrackers();

        if (!supportedTrackers.length) {
            return (
                <Row justify='center' align='middle' style={{ marginTop: '5px' }}>
                    <Col>
                        <Text type='warning' className='cvat-text-color'>
                            No available trackers found
                        </Text>
                    </Col>
                </Row>
            );
        }

        return (
            <>
                <Row justify='start'>
                    <Col>
                        <Text className='cvat-text-color'>Tracker</Text>
                    </Col>
                </Row>
                <Row align='middle' justify='center'>
                    <Col span={24}>
                        <Select
                            style={{ width: '100%' }}
                            defaultValue={supportedTrackers[0].name}
                            onChange={this.setActiveTracker}
                        >
                            {supportedTrackers.map(
                                (tracker: MLModel): JSX.Element => (
                                    <Select.Option value={tracker.id} title={tracker.description} key={tracker.id}>
                                        {tracker.name}
                                    </Select.Option>
                                ),
                            )}
                        </Select>
                    </Col>
                </Row>
                <Row align='middle' justify='end'>
                    <Col>
                        <Button
                            type='primary'
                            loading={fetching}
                            className='cvat-tools-track-button'
                            disabled={!activeTracker || fetching || frame === jobInstance.stopFrame}
                            onClick={() => {
                                if (activeTracker && activeLabelID) {
                                    this.setState({ mode: 'tracking' });

                                    canvasInstance.cancel();
                                    canvasInstance.interact({
                                        shapeType: 'rectangle',
                                        enabled: true,
                                    });

                                    const { onSwitchToolsBlockerState } = this.props;
                                    onInteractionStart(activeTracker, activeLabelID, {});
                                    onSwitchToolsBlockerState({ buttonVisible: false });
                                }
                            }}
                        >
                            Track
                        </Button>
                    </Col>
                </Row>
            </>
        );
    }

    private renderInteractorBlock(): JSX.Element {
        const {
            interactors, canvasInstance, labels, onInteractionStart, frame,
        } = this.props;
        const {
            activeInteractor,
            activeLabelID,
            fetching,
            startInteractingWithBox,
            convertMasksToPolygons,
            maskThreshold,
            logitLoading,
        } = this.state;

        // if (!interactors.length) {
        //     return (
        //         <Row justify='center' align='middle' style={{ marginTop: '5px' }}>
        //             <Col>
        //                 <Text type='warning' className='cvat-text-color'>
        //                     No available interactors found
        //                 </Text>
        //             </Col>
        //         </Row>
        //     );
        // }

        const minNegVertices = activeInteractor?.params?.canvas?.minNegVertices ?? -1;
        const renderStartWithBox = activeInteractor?.params?.canvas?.startWithBoxOptional ?? false;
        const maskAvailable = Boolean(this.interaction.latestResponse.mask?.length) && !logitLoading;
        const thresholdValue = maskThreshold ?? 0.5;

        return (
            <>
                <Row justify='start'>
                    <Col>
                        <Text className='cvat-text-color'>Interactor</Text>
                    </Col>
                </Row>
                <Row align='middle' justify='space-between'>
                    <Col span={22}>
                        <Select
                            style={{ width: '100%' }}
                            defaultValue={interactors.length ? interactors[0].name : 'Mock Interactor (Dev)'}
                            onChange={this.setActiveInteractor}
                        >
                            {interactors.map(
                                (interactor: MLModel): JSX.Element => (
                                    <Select.Option
                                        value={interactor.id}
                                        title={interactor.description}
                                        key={interactor.id}
                                    >
                                        {interactor.name}
                                    </Select.Option>
                                ),
                            )}
                        </Select>
                    </Col>
                    <Col span={2} className='cvat-interactors-tips-icon-container'>
                        <Popover
                            destroyTooltipOnHide
                            content={(
                                <ToolsTooltips
                                    name={activeInteractor?.name}
                                    withNegativePoints={minNegVertices >= 0}
                                    {...(activeInteractor?.tip || {})}
                                />
                            )}
                        >
                            <QuestionCircleOutlined />
                        </Popover>
                    </Col>
                </Row>
                <div className='cvat-tools-interactor-setups'>
                    <div>
                        <Switch
                            checked={convertMasksToPolygons}
                            onChange={(checked: boolean) => {
                                this.setState({ convertMasksToPolygons: checked });
                            }}
                        />
                        <Text>Convert masks to polygons</Text>
                    </div>

                    <div>
                        <Text>Mask threshold</Text>
                        <Row gutter={8} align='middle'>
                            <Col span={16}>
                                <Slider
                                    min={0}
                                    max={1}
                                    step={0.01}
                                    value={thresholdValue}
                                    onChange={this.handleMaskThresholdSliderChange}
                            disabled={!maskAvailable}
                                />
                            </Col>
                            <Col span={8}>
                                <InputNumber
                                    min={0}
                                    max={1}
                                    step={0.01}
                                    value={maskThreshold ?? undefined}
                                    onChange={this.handleMaskThresholdInputChange}
                            disabled={!maskAvailable}
                                    style={{ width: '100%' }}
                                />
                            </Col>
                        </Row>
                    </div>

                    {renderStartWithBox && (
                        <div>
                            <Switch
                                checked={startInteractingWithBox}
                                onChange={(value: boolean) => this.setState({ startInteractingWithBox: value })}
                            />
                            <Text>Start with a bounding box</Text>
                        </div>
                    )}
                </div>
                <Row align='middle' justify='end'>
                    <Col>
                    <Button
    type='primary'
    loading={fetching}
    className='cvat-tools-interact-button'
    disabled={fetching} // Chỉ khóa khi đang tải, không khóa khi thiếu model
    onClick={async () => {
        try {
            console.log("=== INTERACT WITH LOGIT (No Server) ===");

            const { frame, canvasInstance } = this.props;
            const { activeLabelID } = this.state;

            // 1. Validate
            if (!activeLabelID) {
                notification.warning({
                    message: 'Please select a label first',
                });
                return;
            }

            console.log("1. Frame:", frame, "Label:", activeLabelID);

            // 2. Set mode
            this.setState({ mode: 'interaction' });
            console.log("2. Mode set to 'interaction'");

            // 3. ✅ KHÔNG GỌI onInteractionStart - Chỉ cancel canvas cũ
            canvasInstance.cancel();
            console.log("3. Canvas cleared");

            // 4. Initialize interaction state
            this.interaction.id = lodash.uniqueId('interaction_');
            this.interaction.isAborted = false;
            this.interaction.latestResponse = {
                rle: [],
                points: [],
                mask: [[1]], // Placeholder
                threshold: 0.5,
                bounds: [0, 0, 1, 1],
            };
            console.log("4. Interaction state initialized, ID:", this.interaction.id);

            // 5. Load logit image
            console.log("5. Loading logit for frame:", frame);
            await this.loadLogitForFrame(frame);

            console.log("✅ Interact completed successfully!");

        } catch (e) {
            console.error("❌ Error in Interact:", e);
            notification.error({
                message: 'Failed to load logit',
                description: e instanceof Error ? e.message : String(e),
            });
        }
    }}
>
    Interact
</Button>
                    </Col>
                </Row>
            </>
        );
    }

    private renderDetectorBlock(): JSX.Element {
        const {
            jobInstance, detectors, curZOrder, frame, labels, createAnnotations,
        } = this.props;

        if (!detectors.length) {
            return (
                <Row justify='center' align='middle' style={{ marginTop: '5px' }}>
                    <Col>
                        <Text type='warning' className='cvat-text-color'>
                            No available detectors found
                        </Text>
                    </Col>
                </Row>
            );
        }

        return (
            <DetectorRunner
                withCleanup={false}
                models={detectors}
                labels={labels}
                dimension={jobInstance.dimension}
                runInference={async (model: MLModel, body: AnnotateTaskRequestBody) => {
                    function loadAttributes(
                        attributes: { spec_id: number; value: string }[],
                    ): Record<number, string> {
                        return Object.fromEntries(attributes.map((a) => [a.spec_id, a.value]));
                    }

                    try {
                        this.setState({ mode: 'detection', fetching: true });

                        // The function call endpoint doesn't support the cleanup parameter.
                        const { cleanup, ...restOfBody } = body;

                        const result = await core.lambda.call(jobInstance.taskId, model, {
                            ...restOfBody, type: 'annotate_frame', frame, job: jobInstance.id,
                        }) as DetectorResults;

                        const tagStates = result.tags.map((tag) => {
                            const jobLabel = jobInstance.labels
                                .find((jLabel) => jLabel.id === tag.label_id)!;

                            return new core.classes.ObjectState({
                                attributes: loadAttributes(tag.attributes),
                                frame,
                                label: jobLabel,
                                objectType: ObjectType.TAG,
                                source: core.enums.Source.AUTO,
                            });
                        });

                        const shapeStates = result.shapes.map((shape) => {
                            const jobLabel = jobInstance.labels
                                .find((jLabel) => jLabel.id === shape.label_id)!;

                            return new core.classes.ObjectState({
                                attributes: loadAttributes(shape.attributes),
                                elements: shape.elements?.map((element) => {
                                    const jobSublabel = jobLabel.structure!.sublabels
                                        .find((sublabel) => sublabel.id === element.label_id)!;

                                    return {
                                        attributes: loadAttributes(element.attributes),
                                        frame,
                                        label: jobSublabel,
                                        objectType: ObjectType.SHAPE,
                                        occluded: element.occluded,
                                        outside: element.outside,
                                        points: element.points,
                                        shapeType: element.type,
                                        source: core.enums.Source.AUTO,
                                    };
                                }),
                                frame,
                                label: jobLabel,
                                objectType: ObjectType.SHAPE,
                                occluded: shape.occluded,
                                points: shape.points,
                                rotation: shape.rotation,
                                shapeType: shape.type,
                                source: core.enums.Source.AUTO,
                                zOrder: curZOrder,
                            });
                        });

                        createAnnotations([...tagStates, ...shapeStates]);
                    } catch (error: any) {
                        notification.error({
                            description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                            message: 'Detection error occurred',
                            duration: null,
                        });
                    } finally {
                        this.setState({ fetching: false });
                    }
                }}
            />
        );
    }

    private renderPopoverContent(): JSX.Element {
        return (
            <div className='cvat-tools-control-popover-content'>
                <Row justify='start'>
                    <Col>
                        <Text className='cvat-text-color' strong>
                            AI Tools
                        </Text>
                    </Col>
                </Row>
                <Tabs
                    type='card'
                    tabBarGutter={8}
                    items={[{
                        key: 'interactors',
                        label: 'Interactors',
                        children: (
                            <>
                                {this.renderLabelBlock()}
                                {this.renderInteractorBlock()}
                            </>
                        ),
                    }, {
                        key: 'detectors',
                        label: 'Detectors',
                        children: this.renderDetectorBlock(),
                    }, {
                        key: 'trackers',
                        label: 'Trackers',
                        children: (
                            <>
                                {this.renderLabelBlock()}
                                {this.renderTrackerBlock()}
                            </>
                        ),
                    }]}
                />
            </div>
        );
    }

    public render(): JSX.Element | null {
        const {
            interactors, detectors, trackers, isActivated, canvasInstance, labels, frameIsDeleted,
        } = this.props;
        const {
            fetching, approxPolyAccuracy, pointsReceived, mode, portals, convertMasksToPolygons,
        } = this.state;

        // if (![...interactors, ...detectors, ...trackers].length) return null;

        const dynamicPopoverProps = isActivated ?
            {
                overlayStyle: {
                    display: 'none',
                },
            } :
            {};

        const dynamicIconProps = isActivated ?
            {
                className: 'cvat-tools-control cvat-active-canvas-control',
                onClick: (): void => {
                    canvasInstance.interact({ enabled: false });
                },
            } :
            {
                className: 'cvat-tools-control',
            };

        const showAnyContent = labels.length && !frameIsDeleted;
        const showInteractionContent = isActivated && mode === 'interaction' && pointsReceived && convertMasksToPolygons;
        const showDetectionContent = fetching && mode === 'detection';

        const interactionContent: JSX.Element | null = showInteractionContent ? (
            <ApproximationAccuracy
                approxPolyAccuracy={approxPolyAccuracy}
                onChange={(value: number) => {
                    this.setState({ approxPolyAccuracy: value });
                }}
            />
        ) : null;

        const detectionContent: JSX.Element | null = showDetectionContent ? (
            <Modal
                title='Making a server request'
                zIndex={Number.MAX_SAFE_INTEGER}
                open
                destroyOnClose
                closable={false}
                footer={[]}
            >
                <Text>Waiting for a server response..</Text>
                <LoadingOutlined style={{ marginLeft: '10px' }} />
            </Modal>
        ) : null;

        return showAnyContent ? (
            <>
                <CustomPopover {...dynamicPopoverProps} placement='right' content={this.renderPopoverContent()}>
                    <Icon {...dynamicIconProps} component={AIToolsIcon} />
                </CustomPopover>
                {interactionContent}
                {detectionContent}
                {portals}
            </>
        ) : (
            <Icon className=' cvat-tools-control cvat-disabled-canvas-control' component={AIToolsIcon} />
        );
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(ToolsControlComponent);
