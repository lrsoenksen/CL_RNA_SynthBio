import unittest
import rna


class TestMoleculeClass(unittest.TestCase):
    def test_constructor_no_bracket(self):
        seq = 'AUGC'
        molecule = rna.Molecule(seq)
        self.assertEqual(molecule.seq, seq)

    def test_constructor_incorrect_seq(self):
        seq = 'AUGB'
        with self.assertRaises(Exception):
            rna.Molecule(seq)

    def test_constructor_with_bracket(self):
        seq = 'AUGC'
        dot = '(..)'
        molecule = rna.Molecule(seq, dot)
        self.assertEqual(molecule.seq, seq)
        self.assertEqual(len(molecule.seq), len(molecule.dot))

    def test_constructor_with_incorrect_bracket(self):
        seq = 'AUGC'
        dot = '(...'
        with self.assertRaises(Exception):
            rna.Molecule(seq, dot)

    def test_constructor_with_incorrect_bracket2(self):
        seq = 'AUGC'
        dot = ')..('
        with self.assertRaises(Exception):
            rna.Molecule(seq, dot)

    def test_show_method(self):
        m = rna.Molecule('AGGCU')
        with self.assertRaises(Exception):
            m.show()

    def test_substrings_method(self):
        m = rna.Molecule("GGCCUGAGGAGACUCAGAAGCC", "((((((((....)))))..)))")
        sub = m.get_substrings(6)
        self.assertEqual(type(sub), list)
        self.assertEqual(len(sub), 1)
        self.assertEqual(sub[0].dot, '(....)')
        self.assertEqual(len(m.get_substrings(3)), 2)

    def test_repair_method(self):
        m = rna.Molecule("GGCCUGAGGAGACUCAGAAGCA", "(((((((((..))))))..)))")
        m.repair()
        self.assertEqual(m.dot, ".(((((((....)))))..)).")

    def test_evaluate_method(self):
        m = rna.Molecule("GGCCUGAGGAGACUCAGAAGCC", "((((((((....)))))..)))")
        k = rna.Molecule("GGCCUGAGGAGACUCAGAAGCC", ".(((((((....)))))..)).")
        l = rna.Molecule("GGCCUGAGGAGACUCAGAAGCC", "(((.(..((....))..).)))")
        self.assertGreater(m.evaluate(), k.evaluate())
        self.assertGreater(m.evaluate(), l.evaluate())


class TestRnaModuleFunctions(unittest.TestCase):
    def test_complementary_function(self):
        self.assertEqual(rna.complementary('A'), 'U')
        self.assertEqual(rna.complementary('C'), 'G')
        self.assertEqual(rna.complementary('G'), 'C')
        self.assertEqual(rna.complementary('U'), 'A')

    def test_is_pair_allowed_function(self):
        self.assertTrue(rna.is_pair_allowed('G', 'U'))
        self.assertFalse(rna.is_pair_allowed('A', 'C'))

    def test_encode_rna_function(self):
        e = rna.encode_rna('GAACGU')
        self.assertEqual(type(e), list)
        self.assertEqual(len(e), 6)
        self.assertTrue(0 in e and 1 in e and 2 in e and 3 in e)

    def test_match_parentheses_function(self):
        m = rna.Molecule("GGCCUGAGGAGACUCAGAAGCC", "((((((((....)))))..)))")
        self.assertEqual(rna.match_parentheses(m.dot, 3), 16)

    def test_dot_reverse_function(self):
        m = rna.Molecule("GGCCUGAGGAGACUCAGAAGCC", "((((((((....)))))..)))")
        rev = rna.dot_reverse(m.dot)
        self.assertEqual(rev[:4], '(((.')
        self.assertEqual(rev.count('('), rev.count(')'))

    def test_pair_matrix_funtion(self):
        m = rna.Molecule("GGCCUGAGGAGACUCAGAAGCC", "((((((((....)))))..)))")
        p = rna.pair_matrix(m)
        self.assertEqual(p.sum(), 16)
        self.assertEqual(p.all(), p.T.all())

    def test_complementarity_matrix_funtion(self):
        m = rna.Molecule("GGCCUGAGGAGACUCAGAAGCC", "((((((((....)))))..)))")
        p = rna.complementarity_matrix(m)
        self.assertEqual(p[0, len(m.seq) - 1], 2)
        self.assertEqual(p[0, len(m.seq) - 2], 2)
        self.assertEqual(p[0, len(m.seq) - 3], 0)
        self.assertEqual(p[m.seq.find('G'), m.seq.find('U')], 1)
        self.assertEqual(p.all(), p.T.all())


if __name__ == '__main__':
    unittest.main()
