

(define (problem BW-rand-3)
(:domain blocksworld-4ops)
(:objects a b c )
(:init
(handempty)
(ontable a)
(on b c)
(on c a)
(clear b)
)
(:goal
	(and
		(on a b)
		(on c a)
		(on b c)
	)
)
)