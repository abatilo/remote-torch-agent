import time
import os
import signal
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
from torch.distributed.elastic.multiprocessing.api import Std
from torch.distributed.elastic.rendezvous.api import RendezvousParameters
from torch.distributed.elastic.metrics.api import prof
from torch.distributed.elastic.agent.server.api import (
    SimpleElasticAgent,
    WorkerSpec,
    WorkerGroup,
    RunResult,
    WorkerState,
)
from kubernetes import config
from kubernetes.client import (
    V1Pod,
    V1ObjectMeta,
    V1Container,
    V1PodSpec,
    CoreV1Api,
)
from typing import Dict, Any


def entrypoint(args):
    # Skipping the formal entrypoint as a hack for the demo
    pass


def pod_name_util(agent_rank: int, local_rank: int) -> str:
    return f"agent-{agent_rank}-worker-{local_rank}"


class KubernetesElasticAgent(SimpleElasticAgent):
    """
    Decoupling elastic worker state and the actual training process
    state. I used Kubernetes as an example but this could be slurm as well
    """

    def __init__(
        self,
        agent_rank: int,
        spec: WorkerSpec,
        exit_barrier_timeout: float = 300,
    ):
        self._agent_rank = agent_rank
        self._kubernetes_client = CoreV1Api()
        super().__init__(spec, exit_barrier_timeout)

    @prof
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        workers = {}
        for worker in worker_group.workers:
            pod_name = pod_name_util(self._agent_rank, worker.local_rank)

            container = V1Container(
                name="ubuntu-container",
                image="ubuntu:latest",
                command=["bash"],
                args=[
                    "-c",
                    """
echo "Pretend this is a script that activates training"

for i in {1..20}
do
   echo "step ${i}"
   sleep 5
done
    """,
                ],
            )
            pod_spec = V1PodSpec(
                termination_grace_period_seconds=5,
                containers=[container],
                restart_policy="Never",
            )
            metadata = V1ObjectMeta(name=pod_name)
            pod = V1Pod(api_version="v1", kind="Pod", metadata=metadata, spec=pod_spec)
            self._kubernetes_client.create_namespaced_pod(body=pod, namespace="default")
            workers[worker.local_rank] = pod_name

        return workers

    @prof
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        pod_statuses = {}
        for worker in worker_group.workers:
            try:
                pod_name = pod_name_util(self._agent_rank, worker.local_rank)
                pod_status = self._kubernetes_client.read_namespaced_pod_status(
                    pod_name, namespace="default"
                )
                pod_statuses[worker.global_rank] = pod_status.status.phase
            except:
                pod_statuses[worker.global_rank] = "Failed"

        if all(pod_status == "Succeeded" for pod_status in pod_statuses.values()):
            return RunResult(
                state=WorkerState.SUCCEEDED,
                return_values=pod_statuses,
            )
        elif any(pod_status == "Failed" for pod_status in pod_statuses.values()):
            return RunResult(
                state=WorkerState.FAILED,
                failures=pod_statuses,
            )

        return RunResult(state=WorkerState.HEALTHY)

    @prof
    def _shutdown(self, death_sig: signal.Signals = signal.SIGTERM) -> None:
        for worker in self._worker_group.workers:
            pod_name = pod_name_util(self._agent_rank, worker.local_rank)
            try:
                self._kubernetes_client.delete_namespaced_pod(
                    pod_name, namespace="default"
                )
            except:
                print(f"Failed to delete pod {pod_name}")

    @prof
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        self._shutdown()


if __name__ == "__main__":
    rank = int(os.environ.get("RANK", 0))

    rdzv_parameters = RendezvousParameters(
        backend="c10d",
        endpoint="localhost:29500",
        run_id="local",
        min_nodes=2,
        max_nodes=4,
    )
    rdzv_handler = rdzv_registry.get_rendezvous_handler(rdzv_parameters)

    spec = WorkerSpec(
        role="trainer",
        local_world_size=1,
        entrypoint=entrypoint,
        args=(),
        rdzv_handler=rdzv_handler,
        master_addr="localhost",
        master_port=29500,
        max_restarts=3,
        tee=Std.ALL,
        monitor_interval=6,
    )

    # Authenticate to k8s with my local ~/.kube/config
    config.load_kube_config()

    agent = KubernetesElasticAgent(
        agent_rank=rank,
        spec=spec,
        exit_barrier_timeout=60,
    )

    agent.run()
