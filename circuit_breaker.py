import time
import logging
from typing import Dict, Any, Callable, TypeVar, Generic, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("circuit_breaker")

# Type variables for generic function handling
T = TypeVar('T')
R = TypeVar('R')

class CircuitBreaker(Generic[T, R]):
    """
    Circuit breaker pattern implementation to handle API failures

    Tracks failures and temporarily disables services that are failing consistently
    to prevent cascading failures and allow services time to recover.
    """
    def __init__(
        self, 
        service_name: str,
        failure_threshold: int = 5,
        cooldown_period: int = 300,  # 5 minutes in seconds
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.cooldown_period = cooldown_period
        self.failures = 0
        self.last_failure_time = 0
        self.is_open = False  # Open circuit = service disabled

    def __call__(
        self, 
        func: Callable[[T], R], 
        *args: Any, 
        fallback: Optional[Callable[[], R]] = None,
        fallback_value: Optional[R] = None,
        **kwargs: Any
    ) -> R:
        """
        Execute the function with circuit breaker protection

        Args:
            func: The function to execute
            *args: Arguments to pass to the function
            fallback: Optional fallback function to call if circuit is open
            fallback_value: Optional fallback value to return if circuit is open
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result of the function or fallback
        """
        # Check if circuit is already open (service disabled)
        if self.is_open:
            # Check if cooldown period has passed
            if time.time() - self.last_failure_time > self.cooldown_period:
                logger.info(f"Circuit breaker for {self.service_name} reset after cooldown")
                self.reset()
            else:
                logger.warning(f"Circuit for {self.service_name} is open, using fallback")
                if fallback:
                    return fallback()
                return fallback_value

        # Circuit is closed, try to execute function
        try:
            result = func(*args, **kwargs)
            self.success()  # Register successful call
            return result
        except Exception as e:
            logger.error(f"Error in {self.service_name}: {e}")
            self.failure()  # Register failure

            # Check if threshold reached after this failure
            if self.is_open:
                logger.warning(f"Circuit breaker for {self.service_name} opened after {self.failures} failures")
                if fallback:
                    return fallback()
                return fallback_value
            else:
                # Re-raise the exception if circuit still closed
                raise

    def success(self) -> None:
        """Register a successful execution"""
        # Reset failure count on success
        if self.failures > 0:
            self.failures = 0
            logger.info(f"Circuit breaker for {self.service_name} failure count reset")

    def failure(self) -> None:
        """Register a failure execution"""
        self.failures += 1
        self.last_failure_time = time.time()

        # Check if we need to open the circuit
        if self.failures >= self.failure_threshold:
            self.is_open = True

    def reset(self) -> None:
        """Reset the circuit breaker state"""
        self.failures = 0
        self.is_open = False
        logger.info(f"Circuit breaker for {self.service_name} reset")

# Global registry of circuit breakers
circuit_breakers: Dict[str, CircuitBreaker] = {}

def get_circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    cooldown_period: int = 300
) -> CircuitBreaker:
    """
    Get or create a circuit breaker for a service

    Args:
        service_name: Name of the service
        failure_threshold: Number of failures before opening circuit
        cooldown_period: Time in seconds to keep circuit open

    Returns:
        Circuit breaker instance
    """
    if service_name not in circuit_breakers:
        circuit_breakers[service_name] = CircuitBreaker(
            service_name=service_name,
            failure_threshold=failure_threshold,
            cooldown_period=cooldown_period
        )
    return circuit_breakers[service_name]

def with_circuit_breaker(
    service_name: str,
    fallback_value: Any = None,
    failure_threshold: int = 5,
    cooldown_period: int = 300
):
    """
    Decorator to apply circuit breaker pattern to a function

    Args:
        service_name: Name of the service
        fallback_value: Value to return if circuit is open
        failure_threshold: Number of failures before opening circuit
        cooldown_period: Time in seconds to keep circuit open

    Returns:
        Decorated function
    """
    circuit_breaker = get_circuit_breaker(
        service_name, 
        failure_threshold, 
        cooldown_period
    )

    def decorator(func):
        def wrapper(*args, **kwargs):
            return circuit_breaker(
                func, 
                *args, 
                fallback_value=fallback_value, 
                **kwargs
            )
        return wrapper
    return decorator