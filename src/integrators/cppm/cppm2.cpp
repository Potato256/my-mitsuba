#include "cppm_framework.h"

// null hypothesis: linear function

MTS_NAMESPACE_BEGIN

namespace {

#define SEC_U 2
#define SEC_V 6
#define SECS (SEC_U * SEC_V)
#define sqr(a) ((a) * (a))
#define cube(a) ((a) * (a) * (a))

using std::min;
using std::max;

struct CPPMGatherPoint: public SPPMFrameworkGatherPoint {
    size_t photonCount[SEC_U * SEC_V];
    Spectrum sumI, sumEmission;
    Float minPhotonCount;
    size_t totalPhotonCount;
    inline CPPMGatherPoint() : SPPMFrameworkGatherPoint(),
        sumI(0.f), sumEmission(0.f), totalPhotonCount(0) {
        memset(photonCount, 0, sizeof(photonCount));
    }
};

struct CPPMQuery {
    CPPMQuery(CPPMGatherPoint &gp, int maxDepth, Sampler *sampler)
        : photonCount(gp.photonCount), its(gp.its), n(gp.its.shFrame.n), position(gp.its.p),
        weight(gp.weight), squaredRadius(gp.radius * gp.radius), maxDepth(maxDepth), sumFlux(0.f),
        sampler(sampler) {
        bsdf = its.getBSDF();
    }
    inline int getSection(const Point &photonPos) {
        Vector vector = its.toLocal(photonPos - position);
        Float angle = atan2(vector.y, vector.x);
        angle += vector.y < 0 || (vector.y == 0 && vector.x < 0) ? 2 * M_PI : 0;
        int i = max(0, min(SEC_U - 1, (int)(SEC_U * vector.lengthSquared() / squaredRadius)));
        int j = (int)(SEC_V * angle / (M_PI * 2)) % SEC_V;
        return i * SEC_V + j;
    }
    inline void operator()(const Photon &photon) {
        Normal photonNormal(photon.getNormal());
        Vector wi = -photon.getDirection();
        Float wiDotGeoN = absDot(photonNormal, wi);
        if (photon.getDepth() > maxDepth
            || dot(photonNormal, n) < 1e-1f
            || wiDotGeoN < 1e-2f)
            return;

        BSDFSamplingRecord bRec(its, its.toLocal(wi), its.wi, EImportance);

        Spectrum value = photon.getPower() * bsdf->eval(bRec);
        if (value.isZero())
            return;
        /* Account for non-symmetry due to shading normals */
        value *= std::abs(Frame::cosTheta(bRec.wi) /
            (wiDotGeoN * Frame::cosTheta(bRec.wo)));

        value *= weight;

        int sec = getSection(photon.position);
        ++photonCount[sec];
        sumFlux += value;
    }
    size_t *photonCount;
    const Intersection &its;
    const Normal &n;
    const Point &position;
    Spectrum weight;
    const BSDF *bsdf;
    Float squaredRadius;
    int maxDepth;
    Spectrum sumFlux;
    Sampler *sampler;
};

const double chi2_90[] = {100, 2.70554, 4.60517, 6.25139, 7.77944, 9.23636, 10.64464, 12.01704, 13.36157, 14.68366, 15.98718, 17.27501, 18.54935, 19.81193, 21.06414, 22.30713, 23.54183, 24.76904, 25.98942, 27.20357, 28.41198, 29.61509, 30.81328, 32.00690, 33.19624, 34.38159, 35.56317, 36.74122, 37.91592, 39.08747, 40.25602, 41.42174, 42.58475, 43.74518, 44.90316, 46.05879, 47.21217, 48.36341, 49.51258, 50.65977, 51.80506, 52.94851, 54.09020, 55.23019, 56.36854, 57.50530, 58.64054, 59.77429, 60.90661, 62.03754};
const double chi2_95[] = {100, 3.84146, 5.99146, 7.81473, 9.48773, 11.07050, 12.59159, 14.06714, 15.50731, 16.91898, 18.30704, 19.67514, 21.02607, 22.36203, 23.68479, 24.99579, 26.29623, 27.58711, 28.86930, 30.14353, 31.41043, 32.67057, 33.92444, 35.17246, 36.41503, 37.65248, 38.88514, 40.11327, 41.33714, 42.55697, 43.77297, 44.98534, 46.19426, 47.39988, 48.60237, 49.80185, 50.99846, 52.19232, 53.38354, 54.57223, 55.75848, 56.94239, 58.12404, 59.30351, 60.48089, 61.65623, 62.82962, 64.00111, 65.17077, 66.33865};
const double chi2_99[] = {100, 6.63490, 9.21034, 11.34487, 13.27670, 15.08627, 16.81189, 18.47531, 20.09024, 21.66599, 23.20925, 24.72497, 26.21697, 27.68825, 29.14124, 30.57791, 31.99993, 33.40866, 34.80531, 36.19087, 37.56623, 38.93217, 40.28936, 41.63840, 42.97982, 44.31410, 45.64168, 46.96294, 48.27824, 49.58788, 50.89218, 52.19139, 53.48577, 54.77554, 56.06091, 57.34207, 58.61921, 59.89250, 61.16209, 62.42812, 63.69074, 64.95007, 66.20624, 67.45935, 68.70951, 69.95683, 71.20140, 72.44331, 73.68264, 74.91947};

} // namespace

class CPPMIntegrator : public SPPMFramework<CPPMGatherPoint> {
public:
    CPPMIntegrator(const Properties &props) : SPPMFramework(props) {
        m_k = props.getFloat("k", 0.8);
        m_beta = props.getFloat("beta", 1.5625);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
            int sceneResID, int sensorResID, int samplerResID) {
        Integrator::preprocess(scene, queue, job, sceneResID, sensorResID, samplerResID);
        getCoefficient();
        return SPPMFramework<CPPMGatherPoint>::preprocess(scene, queue, job, sceneResID, sensorResID, samplerResID);
    }

    CPPMIntegrator(Stream *stream, InstanceManager *manager)
     : SPPMFramework(stream, manager) { }

    std::string getName() { return "CPPM-linear"; }

    MTS_DECLARE_CLASS()

private:
    void getCoefficient() {
        Float sum_a_denom, sum_b_denom;
        sum_a_denom = sum_b_denom = 0.f;
        Float t1, t2, t3;
        for (int i = 0; i < SEC_U; ++i) {
            Float r1 = std::sqrt((Float)i / SEC_U);
            Float r2 = std::sqrt((Float)(i + 1) / SEC_U);
            for (int j = 0; j < SEC_V; ++j) {
                Float phi1 = j * 2 * M_PI / SEC_V;
                Float phi2 = (j + 1) * 2 * M_PI / SEC_V;
                t1 = (cube(r2) - cube(r1)) * (sin(phi2) - sin(phi1)) / 3.f;
                t2 = (cube(r2) - cube(r1)) * (cos(phi2) - cos(phi1)) / -3.f;
                t3 = (sqr(r2) - sqr(r1)) * (phi2 - phi1) / 2.f;
                sum_a_denom += t1 * t1;
                sum_b_denom += t2 * t2;
                int p = i * SEC_V + j;
                m_p_a_coeff[p] = t1; 
                m_p_b_coeff[p] = t2; 
                m_p_c_coeff[p] = t3;
            }
            m_a_denom[i] = sum_a_denom;
            m_b_denom[i] = sum_b_denom;
        }
    }
    inline void initGatherPoint(CPPMGatherPoint &gp) {
        gp.radius = m_kNN == 0 ? m_initialRadius : 0.f;
        gp.minPhotonCount = m_kNN;
    }
    void chiSquaredTest(size_t *photonCount, bool passed[SEC_U]) {
        Float a_O_numer = 0.f, b_O_numer = 0.f, t1, t2, t3;
        size_t sumO = 0;
        for (int r_ind = 0; r_ind < SEC_U; ++r_ind) {
            Float r = std::sqrt((Float)(r_ind + 1) / SEC_U);
            Float c = 1 / (M_PI * sqr(r));
            for (int j = 0; j < SEC_V; ++j) {
                int p = r_ind * SEC_V + j;
                int O = photonCount[p];
                sumO += O;
                a_O_numer += m_p_a_coeff[p] * O;
                b_O_numer += m_p_b_coeff[p] * O;
            }
            if (sumO == 0) {
                passed[r_ind] = true;
            } else {
                Float a, b;
                a = a_O_numer / (sumO * m_a_denom[r_ind]);
                b = b_O_numer / (sumO * m_b_denom[r_ind]);
                Float len = sqr(a) + sqr(b);
                Float constraint = 1.f / sqr(M_PI * cube(r));
                if (len > constraint) {
                    a /= M_PI * cube(r) * sqrt(len);
                    b /= M_PI * cube(r) * sqrt(len);
                }
                Float V = 0.f;
                for (int i = 0; i <= r_ind; ++i)
                    for (int j = 0; j < SEC_V; ++j) {
                        int pos = i * SEC_V + j;
                        t1 = a * m_p_a_coeff[pos];
                        t2 = b * m_p_b_coeff[pos];
                        t3 = c * m_p_c_coeff[pos];
                        int O = photonCount[pos];
                        Float p = t1 + t2 + t3;
                        Float npi = sumO * p;
                        V += sqr(O - npi) / npi;
                    }
                passed[r_ind] = V < chi2_95[(r_ind + 1) * SEC_V - 3];
            }
        }
    }
    Spectrum updateGatherPoint(CPPMGatherPoint &gp, Sampler *sampler) {
        if (gp.depth != -1) {
            int photonMaxDepth = m_maxDepth == -1 ? INT_MAX : m_maxDepth - gp.depth;
            sampler->generate(gp.pos);
            CPPMQuery query(gp, photonMaxDepth, sampler);
            gp.totalPhotonCount += m_photonMap->executeQuery(gp.its.p, gp.radius, query);
            gp.sumI += query.sumFlux / (M_PI * gp.radius * gp.radius);
            if (gp.totalPhotonCount >= gp.minPhotonCount) {
                bool passed[SEC_U];
                chiSquaredTest(gp.photonCount, passed);
                if (!passed[SEC_U - 1]) {
                    memset(gp.photonCount, 0, sizeof(gp.photonCount));
                    gp.totalPhotonCount = 0;
                    gp.minPhotonCount *= m_beta;
                    bool found = false;
                    for (int i = SEC_U - 1; i >= 0; --i)
                        if (passed[i]) {
                            gp.radius *= std::sqrt((Float)(i + 1) / SEC_U);
                            found = true;
                            break;
                        }
                    if (!found) {
                        gp.radius *= std::sqrt(m_k);
                    }
                }
            }
        }
        gp.sumEmission += gp.emission;
        return gp.sumEmission / m_iteration + gp.sumI / m_totalEmitted;
    }
    Float m_k, m_beta;
    Float m_a_denom[SEC_U], m_b_denom[SEC_U];
    Float m_p_a_coeff[SECS], m_p_b_coeff[SECS], m_p_c_coeff[SECS];
};

MTS_IMPLEMENT_CLASS_S(CPPMIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(CPPMIntegrator, "2 Adapative progressive photon mapper");

MTS_NAMESPACE_END
