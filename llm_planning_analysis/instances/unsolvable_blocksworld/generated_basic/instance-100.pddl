

(define (problem BW-rand-3)
(:domain blocksworld-4ops)
(:objects a b c )
(:init
(handempty)
(on a b)
(ontable b)
(ontable c)
(clear a)
(clear c)
)
(:goal
	(and
		(on a b)
		(on b a)
	)
)
)